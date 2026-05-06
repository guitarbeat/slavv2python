"""
Machine Learning Curator for SLAVV

This module provides ML-based curation of vertices and edges detected by the SLAVV algorithm.
It implements various machine learning approaches for automated quality control and refinement
of vascular network detection results.

Based on the MATLAB MLDeployment.py and MLLibrary.py implementations.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
try:
    from ..models import normalize_pipeline_result
    from ..utils import calculate_path_length
    from ..utils.safe_unpickle import safe_load
    from .automatic_curator import AutomaticCurator
    from .curation_heuristics import choose_edges, choose_vertices, extract_uncurated_info
    from .drews_curator import DrewsCurator
    from .ml_curator_features import compute_local_gradient, feature_importance, in_bounds
    from .ml_curator_io import materialize_model_source
    from .ml_curator_training import load_aggregated_training_data
except ImportError:  # pragma: no cover - fallback for direct execution
    from source.analysis.automatic_curator import AutomaticCurator
    from source.analysis.curation_heuristics import (
        choose_edges,
        choose_vertices,
        extract_uncurated_info,
    )
    from source.analysis.drews_curator import DrewsCurator
    from source.analysis.ml_curator_features import (
        compute_local_gradient,
        feature_importance,
        in_bounds,
    )
    from source.analysis.ml_curator_io import materialize_model_source
    from source.analysis.ml_curator_training import load_aggregated_training_data
    from source.models import normalize_pipeline_result
    from source.utils import calculate_path_length
    from source.utils.safe_unpickle import safe_load

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    "AutomaticCurator",
    "DrewsCurator",
    "MLCurator",
    "choose_edges",
    "choose_vertices",
    "extract_uncurated_info",
]


def _sample_edge_energies(trace: np.ndarray, energy_field: np.ndarray) -> list[float]:
    return [
        energy_field[tuple(point.astype(int))]
        for point in trace
        if in_bounds(point.astype(int), energy_field.shape)
    ]


def _edge_energy_features(trace: np.ndarray, energy_field: np.ndarray) -> list[float]:
    try:
        edge_energies = _sample_edge_energies(trace, energy_field)
    except Exception:
        return [0, 0, 0, 0, 0]
    if not edge_energies:
        return [0, 0, 0, 0, 0]
    return [
        float(np.mean(edge_energies)),
        float(np.std(edge_energies)),
        float(np.min(edge_energies)),
        float(np.max(edge_energies)),
        float(np.median(edge_energies)),
    ]


def _connected_vertex_features(
    start_vertex: Any,
    end_vertex: Any,
    vertex_energies: np.ndarray,
    vertex_radii: Any,
    edge_length: float,
) -> list[float]:
    if start_vertex is not None:
        start_energy = vertex_energies[start_vertex]
        start_radius = vertex_radii[start_vertex] if len(vertex_radii) > start_vertex else 0
    else:
        start_energy = 0
        start_radius = 0
    if end_vertex is not None:
        end_energy = vertex_energies[end_vertex]
        end_radius = vertex_radii[end_vertex] if len(vertex_radii) > end_vertex else 0
    else:
        end_energy = 0
        end_radius = 0
    avg_radius = (start_radius + end_radius) / 2
    length_radius_ratio = edge_length / (avg_radius + 1e-10)
    return [
        start_energy,
        end_energy,
        start_radius,
        end_radius,
        avg_radius,
        length_radius_ratio,
        start_energy - end_energy,
    ]


def _direction_change_features(trace: np.ndarray) -> list[float]:
    if len(trace) <= 2:
        return [0, 0, 0]
    directions = np.diff(trace, axis=0)
    direction_changes = []
    for idx in range(len(directions) - 1):
        dot_product = np.dot(directions[idx], directions[idx + 1])
        norm_product = np.linalg.norm(directions[idx]) * np.linalg.norm(directions[idx + 1])
        if norm_product > 0:
            angle = np.arccos(np.clip(dot_product / norm_product, -1, 1))
            direction_changes.append(angle)
    if not direction_changes:
        return [0, 0, 0]
    return [
        float(np.mean(direction_changes)),
        float(np.std(direction_changes)),
        float(np.max(direction_changes)),
    ]


class MLCurator:
    """
    Machine Learning Curator for automated vertex and edge curation
    """

    def __init__(self):
        self.vertex_classifier = None
        self.edge_classifier = None
        self.vertex_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        # Track whether classifiers have been trained
        self.vertex_trained = False
        self.edge_trained = False

    def extract_vertex_features(
        self, vertices: dict[str, Any], energy_data: dict[str, Any], image_shape: tuple[int, ...]
    ) -> np.ndarray:
        """
        Extract features for vertex classification

        Features include:
        - Energy value and statistics
        - Scale information
        - Local neighborhood properties
        - Spatial position features
        """
        positions = vertices["positions"]
        energies = vertices["energies"]
        scales = vertices["scales"]
        radii = vertices.get("radii_pixels", vertices.get("radii", []))
        energy_field = energy_data["energy"]

        n_vertices = len(positions)
        features = []

        for i in range(n_vertices):
            pos = positions[i]
            energy = energies[i]
            scale = scales[i]
            radius = radii[i]

            # Basic features
            vertex_features = [
                energy,
                scale,
                radius,
                radius / (scale + 1e-10),
                *[
                    pos[0] / image_shape[0],  # Normalized Y position
                    pos[1] / image_shape[1],  # Normalized X position
                    pos[2] / image_shape[2],  # Normalized Z position
                ],
            ]

            # Distance from image center
            center = np.array(image_shape) / 2
            dist_from_center = np.linalg.norm(pos - center) / np.linalg.norm(center)
            vertex_features.append(dist_from_center)

            # Local energy statistics
            try:
                # Extract local neighborhood around vertex
                y, x, z = pos.astype(int)
                neighborhood_size = max(1, int(radius))

                y_min = max(0, y - neighborhood_size)
                y_max = min(image_shape[0], y + neighborhood_size + 1)
                x_min = max(0, x - neighborhood_size)
                x_max = min(image_shape[1], x + neighborhood_size + 1)
                z_min = max(0, z - neighborhood_size)
                z_max = min(image_shape[2], z + neighborhood_size + 1)

                local_energy = energy_field[y_min:y_max, x_min:x_max, z_min:z_max]

                if local_energy.size > 0:
                    local_mean = np.mean(local_energy)
                    local_std = np.std(local_energy)
                    local_min = np.min(local_energy)
                    local_max = np.max(local_energy)
                    local_median = np.median(local_energy)
                    energy_ratio = energy / (local_mean + 1e-10)
                    vertex_features.extend(
                        [
                            local_mean,
                            local_std,
                            local_min,
                            local_max,
                            local_median,
                            energy_ratio,
                        ]
                    )
                else:
                    vertex_features.extend([energy, 0, energy, energy, energy, 1.0])

            except (IndexError, ValueError):
                # Fallback if neighborhood extraction fails
                vertex_features.extend([energy, 0, energy, energy, energy, 1.0])

            # Energy gradient features
            try:
                gradient = compute_local_gradient(energy_field, pos)
                gradient_magnitude = np.linalg.norm(gradient)
                vertex_features.extend(
                    [
                        gradient_magnitude,
                        gradient[0],  # Y gradient
                        gradient[1],  # X gradient
                        gradient[2],  # Z gradient
                    ]
                )
            except Exception:
                vertex_features.extend([0, 0, 0, 0])

            features.append(vertex_features)

        return np.array(features)

    def extract_edge_features(
        self, edges: dict[str, Any], vertices: dict[str, Any], energy_data: dict[str, Any]
    ) -> np.ndarray:
        """
        Extract features for edge classification

        Features include:
        - Edge length and tortuosity
        - Energy statistics along edge
        - Connection properties
        - Geometric features
        """
        edge_traces = edges["traces"]
        edge_connections = edges["connections"]
        vertices["positions"]
        vertex_energies = vertices["energies"]
        vertex_radii = vertices.get("radii_pixels", vertices.get("radii", []))
        energy_field = energy_data["energy"]

        features = []

        for _i, (trace, connection) in enumerate(zip(edge_traces, edge_connections)):
            if len(trace) < 2:
                continue

            trace = np.array(trace)
            start_vertex, end_vertex = connection

            # Basic geometric features
            edge_length = calculate_path_length(trace)
            euclidean_distance = np.linalg.norm(trace[-1] - trace[0])
            tortuosity = edge_length / (euclidean_distance + 1e-10)

            edge_features = [
                edge_length,
                euclidean_distance,
                tortuosity,
                len(trace),  # Number of points in trace
            ]

            edge_features.extend(_edge_energy_features(trace, energy_field))
            edge_features.extend(
                _connected_vertex_features(
                    start_vertex,
                    end_vertex,
                    vertex_energies,
                    vertex_radii,
                    edge_length,
                )
            )
            edge_features.extend(_direction_change_features(trace))

            features.append(edge_features)

        return np.array(features)

    def train_vertex_classifier(
        self, features: np.ndarray, labels: np.ndarray, method: str = "single_hidden_layer_mlp"
    ) -> dict[str, Any]:
        """Train vertex classifier using provided features and labels.

        Args:
            features: Feature matrix (``n_samples``, ``n_features``)
            labels: Binary labels (1 for true vertex, 0 for false positive)
            method: Classification method ('random_forest', 'svm', 'neural_network',
                'gradient_boosting', 'single_hidden_layer_mlp')

        Returns:
            Training results and performance metrics
        """
        logger.info(f"Training vertex classifier using {method}")

        # Scale features
        features_scaled = self.vertex_scaler.fit_transform(features)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Initialize classifier
        if method == "random_forest":
            self.vertex_classifier = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        elif method == "svm":
            self.vertex_classifier = SVC(kernel="rbf", probability=True, random_state=42)
        elif method == "neural_network":
            self.vertex_classifier = MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000
            )
        elif method == "single_hidden_layer_mlp":
            # Compact single-hidden-layer MLP for lightweight curation training.
            self.vertex_classifier = MLPClassifier(
                hidden_layer_sizes=(16,),
                activation="logistic",
                solver="lbfgs",
                max_iter=500,
                random_state=42,
            )
        elif method == "gradient_boosting":
            self.vertex_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Train classifier
        self.vertex_classifier.fit(X_train, y_train)

        # Evaluate performance
        train_score = self.vertex_classifier.score(X_train, y_train)
        test_score = self.vertex_classifier.score(X_test, y_test)

        # Cross-validation
        cv_scores = cross_val_score(self.vertex_classifier, features_scaled, labels, cv=5)

        # Predictions for detailed metrics
        y_pred = self.vertex_classifier.predict(X_test)
        self.vertex_classifier.predict_proba(X_test)[:, 1]

        results = {
            "method": method,
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "feature_importance": feature_importance(self.vertex_classifier),
            "n_features": features.shape[1],
            "n_samples": features.shape[0],
        }

        logger.info(f"Vertex classifier trained. Test accuracy: {test_score:.3f}")
        self.vertex_trained = True
        return results

    def train_edge_classifier(
        self, features: np.ndarray, labels: np.ndarray, method: str = "single_hidden_layer_mlp"
    ) -> dict[str, Any]:
        """Train edge classifier using provided features and labels.

        Args:
            features: Feature matrix (``n_samples``, ``n_features``)
            labels: Binary labels (1 for true edge, 0 for false positive)
            method: Classification method ('random_forest', 'svm', 'neural_network',
                'gradient_boosting', 'single_hidden_layer_mlp')

        Returns:
            Training results and performance metrics
        """
        logger.info(f"Training edge classifier using {method}")

        # Scale features
        features_scaled = self.edge_scaler.fit_transform(features)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Initialize classifier
        if method == "random_forest":
            self.edge_classifier = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        elif method == "svm":
            self.edge_classifier = SVC(kernel="rbf", probability=True, random_state=42)
        elif method == "neural_network":
            self.edge_classifier = MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000
            )
        elif method == "single_hidden_layer_mlp":
            # Compact single-hidden-layer MLP for lightweight curation training.
            self.edge_classifier = MLPClassifier(
                hidden_layer_sizes=(32,),
                activation="logistic",
                solver="lbfgs",
                max_iter=500,
                random_state=42,
            )
        elif method == "gradient_boosting":
            self.edge_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Train classifier
        self.edge_classifier.fit(X_train, y_train)

        # Evaluate performance
        train_score = self.edge_classifier.score(X_train, y_train)
        test_score = self.edge_classifier.score(X_test, y_test)

        # Cross-validation
        cv_scores = cross_val_score(self.edge_classifier, features_scaled, labels, cv=5)

        # Predictions for detailed metrics
        y_pred = self.edge_classifier.predict(X_test)

        results = {
            "method": method,
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "feature_importance": feature_importance(self.edge_classifier),
            "n_features": features.shape[1],
            "n_samples": features.shape[0],
        }

        logger.info(f"Edge classifier trained. Test accuracy: {test_score:.3f}")
        self.edge_trained = True
        return results

    def curate_vertices(
        self,
        vertices: dict[str, Any],
        energy_data: dict[str, Any],
        image_shape: tuple[int, ...],
        confidence_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Curate vertices using trained classifier

        Args:
            vertices: Vertex data from SLAVV processing
            energy_data: Energy field data
            image_shape: Shape of original image
            confidence_threshold: Minimum confidence for keeping vertices

        Returns:
            Curated vertex data with confidence scores
        """
        if self.vertex_classifier is None:
            raise ValueError("Vertex classifier not trained. Call train_vertex_classifier first.")

        logger.info("Curating vertices using ML classifier")

        # Extract features
        features = self.extract_vertex_features(vertices, energy_data, image_shape)
        features_scaled = self.vertex_scaler.transform(features)

        # Predict probabilities
        probabilities = self.vertex_classifier.predict_proba(features_scaled)[:, 1]
        predictions = probabilities >= confidence_threshold

        # Filter vertices based on predictions
        kept_indices = np.where(predictions)[0]

        curated_vertices = {
            "positions": vertices["positions"][kept_indices],
            "scales": vertices["scales"][kept_indices],
            "energies": vertices["energies"][kept_indices],
            "radii_pixels": vertices.get("radii_pixels", vertices.get("radii", []))[kept_indices],
            "radii_microns": vertices.get("radii_microns", vertices.get("radii", []))[kept_indices],
            "radii": vertices.get("radii_microns", vertices.get("radii", []))[kept_indices],
            "confidence_scores": probabilities[kept_indices],
            "original_indices": kept_indices,
        }

        logger.info(
            f"Vertex curation complete: {len(vertices['positions'])} -> {len(kept_indices)} vertices"
        )

        return curated_vertices

    def curate_edges(
        self,
        edges: dict[str, Any],
        vertices: dict[str, Any],
        energy_data: dict[str, Any],
        confidence_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Curate edges using trained classifier
        """
        if self.edge_classifier is None:
            raise ValueError("Edge classifier not trained. Call train_edge_classifier first.")

        logger.info("Curating edges using ML classifier")

        # Extract features
        features = self.extract_edge_features(edges, vertices, energy_data)
        features_scaled = self.edge_scaler.transform(features)

        # Predict probabilities
        probabilities = self.edge_classifier.predict_proba(features_scaled)[:, 1]
        predictions = probabilities >= confidence_threshold

        # Filter edges based on predictions
        kept_indices = np.where(predictions)[0]

        curated_edges = {
            "traces": [edges["traces"][i] for i in kept_indices],
            "connections": [edges["connections"][i] for i in kept_indices],
            "confidence_scores": probabilities[kept_indices],
            "original_indices": kept_indices,
            "vertex_positions": edges["vertex_positions"],
        }

        logger.info(f"Edge curation complete: {len(edges['traces'])} -> {len(kept_indices)} edges")

        return curated_edges

    def save_models(self, vertex_path: Any | None = None, edge_path: Any | None = None) -> None:
        """Save trained models and scalers.

        Parameters:
            vertex_path: Destination for the vertex classifier (file path or file-like object).
            edge_path: Destination for the edge classifier (file path or file-like object).
        """
        if vertex_path and self.vertex_classifier is not None:
            joblib.dump(
                {"classifier": self.vertex_classifier, "scaler": self.vertex_scaler},
                vertex_path,
            )
            logger.info(f"Vertex model saved to {vertex_path}")

        if edge_path and self.edge_classifier is not None:
            joblib.dump(
                {"classifier": self.edge_classifier, "scaler": self.edge_scaler},
                edge_path,
            )
            logger.info(f"Edge model saved to {edge_path}")

    def load_models(self, vertex_path: Any | None = None, edge_path: Any | None = None) -> None:
        """Load trained models and scalers.

        Parameters:
            vertex_path: Source for the vertex classifier (file path).
            edge_path: Source for the edge classifier (file path).
        """
        if vertex_path:
            try:
                with materialize_model_source(vertex_path) as materialized_vertex_path:
                    vertex_data = safe_load(materialized_vertex_path)
                self.vertex_classifier = vertex_data["classifier"]
                self.vertex_scaler = vertex_data["scaler"]
                logger.info(f"Vertex model loaded from {vertex_path}")
            except FileNotFoundError:
                logger.warning(f"Vertex model not found at {vertex_path}")
            except pickle.UnpicklingError as e:
                logger.error(f"Failed to load vertex model from {vertex_path}: {e}")
                raise
            except (ValueError, EOFError) as e:
                logger.error(f"Failed to load vertex model from {vertex_path}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading vertex model from {vertex_path}: {e}")

        if edge_path:
            try:
                with materialize_model_source(edge_path) as materialized_edge_path:
                    edge_data = safe_load(materialized_edge_path)
                self.edge_classifier = edge_data["classifier"]
                self.edge_scaler = edge_data["scaler"]
                logger.info(f"Edge model loaded from {edge_path}")
            except FileNotFoundError:
                logger.warning(f"Edge model not found at {edge_path}")
            except pickle.UnpicklingError as e:
                logger.error(f"Failed to load edge model from {edge_path}: {e}")
                raise
            except (ValueError, EOFError) as e:
                logger.error(f"Failed to load edge model from {edge_path}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading edge model from {edge_path}: {e}")

    def generate_training_data(
        self, processing_results: list[dict[str, Any]], manual_annotations: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate training data from processing results and manual annotations

        Args:
            processing_results: List of SLAVV processing results
            manual_annotations: List of manual curation annotations

        Returns:
            vertex_features, vertex_labels, edge_features, edge_labels
        """
        vertex_features_list = []
        vertex_labels_list = []
        edge_features_list = []
        edge_labels_list = []

        for results, annot in zip(processing_results, manual_annotations):
            typed_result = normalize_pipeline_result(results)
            normalized_results = typed_result.to_dict()
            image_shape = (
                typed_result.energy_data.image_shape
                if typed_result.energy_data is not None
                else tuple(int(value) for value in results.get("image_shape", (100, 100, 50)))
            )

            # Extract vertex features and labels
            v_features = self.extract_vertex_features(
                normalized_results["vertices"],
                normalized_results["energy_data"],
                image_shape,
            )
            v_labels = annot.get("vertex_labels", np.ones(len(v_features)))

            vertex_features_list.append(v_features)
            vertex_labels_list.append(v_labels)

            # Extract edge features and labels
            e_features = self.extract_edge_features(
                normalized_results["edges"],
                normalized_results["vertices"],
                normalized_results["energy_data"],
            )
            e_labels = annot.get("edge_labels", np.ones(len(e_features)))

            edge_features_list.append(e_features)
            edge_labels_list.append(e_labels)

        # Combine all data
        vertex_features = np.vstack(vertex_features_list) if vertex_features_list else np.array([])
        vertex_labels = np.hstack(vertex_labels_list) if vertex_labels_list else np.array([])
        edge_features = np.vstack(edge_features_list) if edge_features_list else np.array([])
        edge_labels = np.hstack(edge_labels_list) if edge_labels_list else np.array([])

        return vertex_features, vertex_labels, edge_features, edge_labels

    def aggregate_training_data(
        self, data_dir: Any, file_pattern: str = "*_results.json"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate features from multiple result snippets for training."""
        return load_aggregated_training_data(data_dir, file_pattern=file_pattern)
