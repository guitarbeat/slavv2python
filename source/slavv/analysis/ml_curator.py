"""
Machine Learning Curator for SLAVV

This module provides ML-based curation of vertices and edges detected by the SLAVV algorithm.
It implements various machine learning approaches for automated quality control and refinement
of vascular network detection results.

Based on the MATLAB MLDeployment.py and MLLibrary.py implementations.
"""

import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings
import pickle
warnings.filterwarnings('ignore')
try:
    from ..utils import calculate_path_length
    from ..utils.safe_unpickle import safe_load
except ImportError:  # pragma: no cover - fallback for direct execution
    from slavv.utils import calculate_path_length
    from slavv.utils.safe_unpickle import safe_load

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
    def extract_vertex_features(self, vertices: Dict[str, Any], energy_data: Dict[str, Any], 
                               image_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Extract features for vertex classification
        
        Features include:
        - Energy value and statistics
        - Scale information
        - Local neighborhood properties
        - Spatial position features
        """
        positions = vertices['positions']
        energies = vertices['energies']
        scales = vertices['scales']
        radii = vertices.get('radii_pixels', vertices.get('radii', []))
        energy_field = energy_data['energy']
        
        n_vertices = len(positions)
        features = []
        
        for i in range(n_vertices):
            pos = positions[i]
            energy = energies[i]
            scale = scales[i]
            radius = radii[i]
            
            # Basic features
            vertex_features = [
                energy,  # Primary energy value
                scale,   # Scale index
                radius,  # Estimated radius
                radius / (scale + 1e-10),  # Radius-to-scale ratio
            ]
            
            # Spatial features (normalized by image dimensions)
            vertex_features.extend([
                pos[0] / image_shape[0],  # Normalized Y position
                pos[1] / image_shape[1],  # Normalized X position
                pos[2] / image_shape[2],  # Normalized Z position
            ])
            
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
                    vertex_features.extend([
                        local_mean,
                        local_std,
                        local_min,
                        local_max,
                        local_median,
                        energy_ratio,
                    ])
                else:
                    vertex_features.extend([energy, 0, energy, energy, energy, 1.0])

            except (IndexError, ValueError):
                # Fallback if neighborhood extraction fails
                vertex_features.extend([energy, 0, energy, energy, energy, 1.0])
            
            # Energy gradient features
            try:
                gradient = self._compute_local_gradient(energy_field, pos)
                gradient_magnitude = np.linalg.norm(gradient)
                vertex_features.extend([
                    gradient_magnitude,
                    gradient[0],  # Y gradient
                    gradient[1],  # X gradient
                    gradient[2],  # Z gradient
                ])
            except Exception:
                vertex_features.extend([0, 0, 0, 0])
            
            features.append(vertex_features)
        
        return np.array(features)
    
    def extract_edge_features(self, edges: Dict[str, Any], vertices: Dict[str, Any], 
                             energy_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features for edge classification
        
        Features include:
        - Edge length and tortuosity
        - Energy statistics along edge
        - Connection properties
        - Geometric features
        """
        edge_traces = edges['traces']
        edge_connections = edges['connections']
        vertices['positions']
        vertex_energies = vertices['energies']
        vertex_radii = vertices.get('radii_pixels', vertices.get('radii', []))
        energy_field = energy_data['energy']
        
        features = []
        
        for i, (trace, connection) in enumerate(zip(edge_traces, edge_connections)):
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
            
            # Energy statistics along edge
            try:
                edge_energies = []
                for point in trace:
                    pos = point.astype(int)
                    if self._in_bounds(pos, energy_field.shape):
                        edge_energies.append(energy_field[tuple(pos)])
                
                if edge_energies:
                    edge_features.extend([
                        np.mean(edge_energies),
                        np.std(edge_energies),
                        np.min(edge_energies),
                        np.max(edge_energies),
                        np.median(edge_energies)
                    ])
                else:
                    edge_features.extend([0, 0, 0, 0, 0])
            except Exception:
                edge_features.extend([0, 0, 0, 0, 0])
            
            # Vertex connection features
            if start_vertex is not None:
                start_energy = vertex_energies[start_vertex]
                start_radius = (
                    vertex_radii[start_vertex] if len(vertex_radii) > start_vertex else 0
                )
            else:
                start_energy = 0
                start_radius = 0
            if end_vertex is not None:
                end_energy = vertex_energies[end_vertex]
                end_radius = (
                    vertex_radii[end_vertex] if len(vertex_radii) > end_vertex else 0
                )
            else:
                end_energy = 0
                end_radius = 0

            avg_radius = (start_radius + end_radius) / 2
            length_radius_ratio = edge_length / (avg_radius + 1e-10)
            energy_diff = start_energy - end_energy

            edge_features.extend([
                start_energy,
                end_energy,
                start_radius,
                end_radius,
                avg_radius,
                length_radius_ratio,
                energy_diff,
            ])
            
            # Directional consistency
            if len(trace) > 2:
                directions = np.diff(trace, axis=0)
                direction_changes = []
                for j in range(len(directions) - 1):
                    dot_product = np.dot(directions[j], directions[j+1])
                    norm_product = np.linalg.norm(directions[j]) * np.linalg.norm(directions[j+1])
                    if norm_product > 0:
                        angle = np.arccos(np.clip(dot_product / norm_product, -1, 1))
                        direction_changes.append(angle)
                
                if direction_changes:
                    edge_features.extend([
                        np.mean(direction_changes),
                        np.std(direction_changes),
                        np.max(direction_changes)
                    ])
                else:
                    edge_features.extend([0, 0, 0])
            else:
                edge_features.extend([0, 0, 0])
            
            features.append(edge_features)
        
        return np.array(features)
    
    def train_vertex_classifier(
        self, features: np.ndarray, labels: np.ndarray, method: str = 'matlab_nn'
    ) -> Dict[str, Any]:
        """Train vertex classifier using provided features and labels.

        Args:
            features: Feature matrix (``n_samples``, ``n_features``)
            labels: Binary labels (1 for true vertex, 0 for false positive)
            method: Classification method ('random_forest', 'svm', 'neural_network',
                'gradient_boosting', 'matlab_nn')

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
        if method == 'random_forest':
            self.vertex_classifier = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        elif method == 'svm':
            self.vertex_classifier = SVC(
                kernel='rbf', probability=True, random_state=42
            )
        elif method == 'neural_network':
            self.vertex_classifier = MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000
            )
        elif method == 'matlab_nn':
            # Mimic MATLAB vertexCuratorNetwork architecture: single hidden layer
            self.vertex_classifier = MLPClassifier(
                hidden_layer_sizes=(16,),
                activation='logistic',
                solver='lbfgs',
                max_iter=500,
                random_state=42,
            )
        elif method == 'gradient_boosting':
            self.vertex_classifier = GradientBoostingClassifier(
                n_estimators=100, random_state=42
            )
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
            'method': method,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': self._get_feature_importance(),
            'n_features': features.shape[1],
            'n_samples': features.shape[0]
        }

        logger.info(f"Vertex classifier trained. Test accuracy: {test_score:.3f}")
        self.vertex_trained = True
        return results
    
    def train_edge_classifier(
        self, features: np.ndarray, labels: np.ndarray, method: str = 'matlab_nn'
    ) -> Dict[str, Any]:
        """Train edge classifier using provided features and labels.

        Args:
            features: Feature matrix (``n_samples``, ``n_features``)
            labels: Binary labels (1 for true edge, 0 for false positive)
            method: Classification method ('random_forest', 'svm', 'neural_network',
                'gradient_boosting', 'matlab_nn')

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
        if method == 'random_forest':
            self.edge_classifier = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        elif method == 'svm':
            self.edge_classifier = SVC(
                kernel='rbf', probability=True, random_state=42
            )
        elif method == 'neural_network':
            self.edge_classifier = MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000
            )
        elif method == 'matlab_nn':
            # Approximate MATLAB edgeCuratorNetwork: logistic single hidden layer
            self.edge_classifier = MLPClassifier(
                hidden_layer_sizes=(32,),
                activation='logistic',
                solver='lbfgs',
                max_iter=500,
                random_state=42,
            )
        elif method == 'gradient_boosting':
            self.edge_classifier = GradientBoostingClassifier(
                n_estimators=100, random_state=42
            )
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
            'method': method,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': self._get_edge_feature_importance(),
            'n_features': features.shape[1],
            'n_samples': features.shape[0]
        }

        logger.info(f"Edge classifier trained. Test accuracy: {test_score:.3f}")
        self.edge_trained = True
        return results
    
    def curate_vertices(self, vertices: Dict[str, Any], energy_data: Dict[str, Any], 
                       image_shape: Tuple[int, ...], confidence_threshold: float = 0.5) -> Dict[str, Any]:
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
            'positions': vertices['positions'][kept_indices],
            'scales': vertices['scales'][kept_indices],
            'energies': vertices['energies'][kept_indices],
            'radii_pixels': vertices.get('radii_pixels', vertices.get('radii', []))[kept_indices],
            'radii_microns': vertices.get('radii_microns', vertices.get('radii', []))[kept_indices],
            'radii': vertices.get('radii_microns', vertices.get('radii', []))[kept_indices],
            'confidence_scores': probabilities[kept_indices],
            'original_indices': kept_indices
        }
        
        logger.info(f"Vertex curation complete: {len(vertices['positions'])} → {len(kept_indices)} vertices")
        
        return curated_vertices
    
    def curate_edges(self, edges: Dict[str, Any], vertices: Dict[str, Any], 
                    energy_data: Dict[str, Any], confidence_threshold: float = 0.5) -> Dict[str, Any]:
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
            'traces': [edges['traces'][i] for i in kept_indices],
            'connections': [edges['connections'][i] for i in kept_indices],
            'confidence_scores': probabilities[kept_indices],
            'original_indices': kept_indices,
            'vertex_positions': edges['vertex_positions']
        }
        
        logger.info(f"Edge curation complete: {len(edges['traces'])} → {len(kept_indices)} edges")
        
        return curated_edges
    
    def save_models(self, vertex_path: Optional[Any] = None, edge_path: Optional[Any] = None) -> None:
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

    def load_models(self, vertex_path: Optional[Any] = None, edge_path: Optional[Any] = None) -> None:
        """Load trained models and scalers.

        Parameters:
            vertex_path: Source for the vertex classifier (file path).
            edge_path: Source for the edge classifier (file path).
        """
        if vertex_path:
            try:
                vertex_data = safe_load(vertex_path)
                self.vertex_classifier = vertex_data["classifier"]
                self.vertex_scaler = vertex_data["scaler"]
                logger.info(f"Vertex model loaded from {vertex_path}")
            except FileNotFoundError:
                logger.warning(f"Vertex model not found at {vertex_path}")
            except (pickle.UnpicklingError, ValueError, EOFError) as e:
                logger.error(f"Failed to load vertex model from {vertex_path}: {e}")
                # Don't crash, just log error, effectively leaving classifier as None
            except Exception as e:
                logger.error(f"Unexpected error loading vertex model from {vertex_path}: {e}")

        if edge_path:
            try:
                edge_data = safe_load(edge_path)
                self.edge_classifier = edge_data["classifier"]
                self.edge_scaler = edge_data["scaler"]
                logger.info(f"Edge model loaded from {edge_path}")
            except FileNotFoundError:
                logger.warning(f"Edge model not found at {edge_path}")
            except (pickle.UnpicklingError, ValueError, EOFError) as e:
                logger.error(f"Failed to load edge model from {edge_path}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading edge model from {edge_path}: {e}")
    
    def generate_training_data(self, processing_results: List[Dict[str, Any]], 
                              manual_annotations: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        
        for results, annotations in zip(processing_results, manual_annotations):
            # Extract vertex features and labels
            v_features = self.extract_vertex_features(
                results['vertices'], 
                results['energy_data'], 
                results.get('image_shape', (100, 100, 50))
            )
            v_labels = annotations.get('vertex_labels', np.ones(len(v_features)))
            
            vertex_features_list.append(v_features)
            vertex_labels_list.append(v_labels)
            
            # Extract edge features and labels
            e_features = self.extract_edge_features(
                results['edges'], 
                results['vertices'], 
                results['energy_data']
            )
            e_labels = annotations.get('edge_labels', np.ones(len(e_features)))
            
            edge_features_list.append(e_features)
            edge_labels_list.append(e_labels)
        
        # Combine all data
        vertex_features = np.vstack(vertex_features_list) if vertex_features_list else np.array([])
        vertex_labels = np.hstack(vertex_labels_list) if vertex_labels_list else np.array([])
        edge_features = np.vstack(edge_features_list) if edge_features_list else np.array([])
        edge_labels = np.hstack(edge_labels_list) if edge_labels_list else np.array([])
        
        return vertex_features, vertex_labels, edge_features, edge_labels
    
    def _compute_local_gradient(self, energy_field: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Compute local gradient at given position"""
        pos_int = np.round(pos).astype(int)
        gradient = np.zeros(3)
        
        for i in range(3):
            if 0 < pos_int[i] < energy_field.shape[i] - 1:
                pos_plus = pos_int.copy()
                pos_minus = pos_int.copy()
                pos_plus[i] += 1
                pos_minus[i] -= 1
                
                gradient[i] = (energy_field[tuple(pos_plus)] - energy_field[tuple(pos_minus)]) / 2
        
        return gradient
    
    
    def _in_bounds(self, pos: np.ndarray, shape: Tuple[int, ...]) -> bool:
        """Check if position is within bounds"""
        return all(0 <= p < s for p, s in zip(pos, shape))
    
    def _get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from vertex classifier if available."""
        if hasattr(self.vertex_classifier, 'feature_importances_'):
            return self.vertex_classifier.feature_importances_
        return None

    def _get_edge_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from edge classifier if available."""
        if hasattr(self.edge_classifier, 'feature_importances_'):
            return self.edge_classifier.feature_importances_
        return None

    def aggregate_training_data(
        self,
        data_dir: Union[str, Path],
        file_pattern: str = "*_results.json"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate features from multiple result snippets for training.
        
        Corresponds to logic in `getTrainingArray.m`.
        
        Args:
            data_dir: Directory containing results.
            file_pattern: Glob pattern for result files.
            
        Returns:
            Arrays of (v_feat, v_labels, e_feat, e_labels).
        """
        # This is a placeholder for the logic that walks directories
        # and stacks the features arrays (which must be saved alongside results).
        # Since we don't have a standardized "saved feature file" format yet,
        # we assume files contain the feature dicts or arrays directly.
        
        
        
        # Implementation depends on how `generate_training_data` saves its output.
        # Assuming we have saved .npz or .json files with 'vertex_features', etc.
        
        return np.array([]), np.array([]), np.array([]), np.array([])


class DrewsCurator:
    """
    Experimental curator based on legacy 'edge_curator_Drews.m'.
    
    This implements specific heuristic rules used in earlier versions of the pipeline
    for pruning edges based on tortuosity, min-length relative to radius, and flow properties.
    """
    
    def __init__(self):
        pass
        
    def curate(self, edges: Dict[str, Any], vertices: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Drews' curation logic.
        """
        logger.warning("DrewsCurator is not fully implemented. Passing through edges.")
        return edges

class AutomaticCurator:
    """
    Automatic curation using heuristic rules (no ML training required)
    """

    def __init__(
        self,
        vertex_parameters: Optional[Dict[str, Any]] = None,
        edge_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store default heuristic parameters for vertex and edge curation.

        Parameters
        ----------
        vertex_parameters:
            Default thresholds for vertex curation. Supplied values are merged
            with parameters passed to :meth:`curate_vertices_automatic`.
        edge_parameters:
            Default thresholds for edge curation. Supplied values are merged
            with parameters passed to :meth:`curate_edges_automatic`.
        """

        self.vertex_parameters = vertex_parameters or {}
        self.edge_parameters = edge_parameters or {}
    
    def curate_vertices_automatic(self, vertices: Dict[str, Any], energy_data: Dict[str, Any], 
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatic vertex curation using heuristic rules
        """
        logger.info("Performing automatic vertex curation")
        
        positions = vertices['positions']
        energies = vertices['energies']
        scales = vertices['scales']
        radii = vertices.get('radii_pixels', vertices.get('radii', []))

        params = {**self.vertex_parameters, **(parameters or {})}
        
        # Rule-based filtering
        keep_mask = np.ones(len(positions), dtype=bool)
        
        # Rule 1: Energy threshold
        energy_threshold = params.get('vertex_energy_threshold', -0.1)
        keep_mask &= (energies < energy_threshold)
        
        # Rule 2: Minimum radius
        min_radius = params.get('min_vertex_radius', 0.5)
        keep_mask &= (radii > min_radius)
        
        # Rule 3: Distance from image boundaries
        image_shape = energy_data.get('image_shape', (100, 100, 50))
        boundary_margin = params.get('boundary_margin', 5)
        
        for dim in range(3):
            keep_mask &= (positions[:, dim] > boundary_margin)
            keep_mask &= (positions[:, dim] < image_shape[dim] - boundary_margin)
        
        # Rule 4: Local energy contrast
        energy_field = energy_data['energy']
        contrast_threshold = params.get('contrast_threshold', 0.1)
        
        for i, pos in enumerate(positions):
            if not keep_mask[i]:
                continue
                
            try:
                # Check local contrast
                y, x, z = pos.astype(int)
                neighborhood_size = max(1, int(radii[i]))
                
                y_min = max(0, y - neighborhood_size)
                y_max = min(image_shape[0], y + neighborhood_size + 1)
                x_min = max(0, x - neighborhood_size)
                x_max = min(image_shape[1], x + neighborhood_size + 1)
                z_min = max(0, z - neighborhood_size)
                z_max = min(image_shape[2], z + neighborhood_size + 1)
                
                local_energy = energy_field[y_min:y_max, x_min:x_max, z_min:z_max]
                
                if local_energy.size > 0:
                    contrast = np.std(local_energy)
                    if contrast < contrast_threshold:
                        keep_mask[i] = False
                        
            except (IndexError, ValueError):
                keep_mask[i] = False
        
        # Apply filtering
        kept_indices = np.where(keep_mask)[0]
        
        curated_vertices = {
            'positions': positions[kept_indices],
            'scales': scales[kept_indices],
            'energies': energies[kept_indices],
            'radii_pixels': radii[kept_indices],
            'radii_microns': vertices.get('radii_microns', vertices.get('radii', []))[kept_indices],
            'radii': vertices.get('radii_microns', vertices.get('radii', []))[kept_indices],
            'original_indices': kept_indices
        }
        
        logger.info(f"Automatic vertex curation: {len(positions)} → {len(kept_indices)} vertices")
        
        return curated_vertices
    
    def curate_edges_automatic(self, edges: Dict[str, Any], vertices: Dict[str, Any],
                              parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatic edge curation using heuristic rules
        """
        logger.info("Performing automatic edge curation")
        
        edge_traces = edges['traces']
        edge_connections = edges['connections']

        params = {**self.edge_parameters, **(parameters or {})}
        
        keep_mask = np.ones(len(edge_traces), dtype=bool)
        
        # Rule 1: Minimum edge length
        min_length = params.get('min_edge_length', 2.0)
        
        for i, trace in enumerate(edge_traces):
            if len(trace) < 2:
                keep_mask[i] = False
                continue
                
            trace = np.array(trace)
            edge_length = calculate_path_length(trace)
            
            if edge_length < min_length:
                keep_mask[i] = False
        
        # Rule 2: Maximum tortuosity
        max_tortuosity = params.get('max_edge_tortuosity', 3.0)
        
        for i, trace in enumerate(edge_traces):
            if not keep_mask[i] or len(trace) < 2:
                continue
                
            trace = np.array(trace)
            edge_length = calculate_path_length(trace)
            euclidean_distance = np.linalg.norm(trace[-1] - trace[0])
            
            if euclidean_distance > 0:
                tortuosity = edge_length / euclidean_distance
                if tortuosity > max_tortuosity:
                    keep_mask[i] = False
        
        # Rule 3: Valid connections
        vertex_positions = vertices['positions']
        max_connection_distance = params.get('max_connection_distance', 5.0)

        # Create a mapping from original vertex indices to curated vertex indices
        original_indices = vertices.get("original_indices")
        if original_indices is None:
            # If not provided, assume a 1-to-1 mapping
            original_indices = np.arange(len(vertices["positions"]))

        original_to_curated_idx = {
            orig_idx: i for i, orig_idx in enumerate(original_indices)
        }
        
        for i, (trace, connection) in enumerate(zip(edge_traces, edge_connections)):
            if not keep_mask[i]:
                continue
                
            start_vertex, end_vertex = connection
            
            # Check start connection
            if start_vertex is not None:
                if start_vertex not in original_to_curated_idx:
                    keep_mask[i] = False
                    continue
                curated_start_idx = original_to_curated_idx[start_vertex]
                start_pos = vertex_positions[curated_start_idx]
                trace_start = np.array(trace[0])
                if np.linalg.norm(start_pos - trace_start) > max_connection_distance:
                    keep_mask[i] = False
                    continue
            
            # Check end connection
            if end_vertex is not None:
                if end_vertex not in original_to_curated_idx:
                    keep_mask[i] = False
                    continue
                curated_end_idx = original_to_curated_idx[end_vertex]
                end_pos = vertex_positions[curated_end_idx]
                trace_end = np.array(trace[-1])
                if np.linalg.norm(end_pos - trace_end) > max_connection_distance:
                    keep_mask[i] = False
        
        # Apply filtering
        kept_indices = np.where(keep_mask)[0]
        
        curated_edges = {
            'traces': [edge_traces[i] for i in kept_indices],
            'connections': [edge_connections[i] for i in kept_indices],
            'original_indices': kept_indices,
            'vertex_positions': edges['vertex_positions']
        }
        
        logger.info(f"Automatic edge curation: {len(edge_traces)} → {len(kept_indices)} edges")

        return curated_edges


def choose_vertices(vertices: Dict[str, Any], min_energy: float = 0.0,
                    min_radius: float = 0.0, energy_sign: float = -1.0) -> np.ndarray:
    """Select vertex indices meeting energy and radius thresholds.

    Parameters
    ----------
    vertices:
        Dictionary containing vertex ``energies`` and either ``radii_microns``
        or ``radii_pixels``.
    min_energy:
        Minimum energy magnitude after applying ``energy_sign``. Higher values
        are retained.
    min_radius:
        Minimum allowed vertex radius in microns.
    energy_sign:
        Sign convention for vessel energy; default ``-1`` treats energy minima
        as positive confidence.

    Returns
    -------
    numpy.ndarray
        Indices of vertices passing the heuristics.
    """

    energies = vertices['energies'] * energy_sign
    radii = vertices.get('radii_microns', vertices.get('radii_pixels'))
    radii = np.asarray(radii, dtype=float)
    mask = (energies >= min_energy) & (radii >= min_radius)
    return np.flatnonzero(mask)


def choose_edges(edges: Dict[str, Any], min_energy: float = 0.0,
                 min_length: float = 0.0, energy_sign: float = -1.0) -> np.ndarray:
    """Select edge indices meeting energy and length thresholds.

    Parameters
    ----------
    edges:
        Dictionary containing ``traces`` for each edge and ``energies`` giving
        the mean energy along each trace.
    min_energy:
        Minimum mean edge energy after applying ``energy_sign``.
    min_length:
        Minimum physical length of the edge trace (in voxel units).
    energy_sign:
        Sign convention for vessel energy; default ``-1`` treats energy minima
        as positive confidence.

    Returns
    -------
    numpy.ndarray
        Indices of edges passing the heuristics.
    """

    energies = edges['energies'] * energy_sign
    lengths = np.array([calculate_path_length(trace) for trace in edges['traces']])
    mask = (energies >= min_energy) & (lengths >= min_length)
    return np.flatnonzero(mask)


def extract_uncurated_info(
    vertices: Dict[str, Any],
    edges: Dict[str, Any],
    energy_data: Dict[str, Any],
    image_shape: Tuple[int, ...],
) -> Dict[str, np.ndarray]:
    """Extract vertex and edge feature arrays without classification.

    Mirrors MATLAB's ``uncuratedInfoExtractor.m`` by deriving feature sets for
    quality-assurance datasets before any ML-based curation.

    Parameters
    ----------
    vertices:
        Dictionary containing vertex ``positions``, ``energies``, ``scales``, and
        optional ``radii_pixels``.
    edges:
        Dictionary with edge ``traces`` and ``connections``.
    energy_data:
        Dictionary providing the ``energy`` field used for feature extraction.
    image_shape:
        Shape of the original image volume, used for normalized coordinates.

    Returns
    -------
    dict
        ``{"vertex_features": ..., "edge_features": ...}`` feature arrays.
    """

    curator = MLCurator()
    vertex_features = curator.extract_vertex_features(vertices, energy_data, image_shape)
    edge_features = curator.extract_edge_features(edges, vertices, energy_data)
    return {"vertex_features": vertex_features, "edge_features": edge_features}
