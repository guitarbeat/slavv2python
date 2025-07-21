"""
Machine Learning Curator for SLAVV

This module provides ML-based curation of vertices and edges detected by the SLAVV algorithm.
It implements various machine learning approaches for automated quality control and refinement
of vascular network detection results.

Based on the MATLAB MLDeployment.py and MLLibrary.py implementations.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

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
        self.is_trained = False
        
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
        radii = vertices['radii']
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
                    vertex_features.extend([
                        np.mean(local_energy),
                        np.std(local_energy),
                        np.min(local_energy),
                        np.max(local_energy),
                        np.median(local_energy)
                    ])
                else:
                    vertex_features.extend([energy, 0, energy, energy, energy])
                    
            except (IndexError, ValueError):
                # Fallback if neighborhood extraction fails
                vertex_features.extend([energy, 0, energy, energy, energy])
            
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
            except:
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
        vertex_positions = vertices['positions']
        vertex_energies = vertices['energies']
        energy_field = energy_data['energy']
        
        features = []
        
        for i, (trace, connection) in enumerate(zip(edge_traces, edge_connections)):
            if len(trace) < 2:
                continue
                
            trace = np.array(trace)
            start_vertex, end_vertex = connection
            
            # Basic geometric features
            edge_length = self._calculate_path_length(trace)
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
            except:
                edge_features.extend([0, 0, 0, 0, 0])
            
            # Vertex connection features
            if start_vertex is not None:
                start_energy = vertex_energies[start_vertex]
                edge_features.append(start_energy)
            else:
                edge_features.append(0)
                
            if end_vertex is not None:
                end_energy = vertex_energies[end_vertex]
                edge_features.append(end_energy)
            else:
                edge_features.append(0)
            
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
    
    def train_vertex_classifier(self, features: np.ndarray, labels: np.ndarray, 
                               method: str = 'random_forest') -> Dict[str, Any]:
        """
        Train vertex classifier using provided features and labels
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Binary labels (1 for true vertex, 0 for false positive)
            method: Classification method ('random_forest', 'svm', 'neural_network', 'gradient_boosting')
        
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
        y_pred_proba = self.vertex_classifier.predict_proba(X_test)[:, 1]
        
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
        return results
    
    def train_edge_classifier(self, features: np.ndarray, labels: np.ndarray, 
                             method: str = 'random_forest') -> Dict[str, Any]:
        """
        Train edge classifier using provided features and labels
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
            'radii': vertices['radii'][kept_indices],
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
    
    def save_models(self, vertex_path: str, edge_path: str):
        """Save trained models to disk"""
        if self.vertex_classifier is not None:
            joblib.dump({
                'classifier': self.vertex_classifier,
                'scaler': self.vertex_scaler
            }, vertex_path)
            logger.info(f"Vertex model saved to {vertex_path}")
        
        if self.edge_classifier is not None:
            joblib.dump({
                'classifier': self.edge_classifier,
                'scaler': self.edge_scaler
            }, edge_path)
            logger.info(f"Edge model saved to {edge_path}")
    
    def load_models(self, vertex_path: str, edge_path: str):
        """Load trained models from disk"""
        try:
            vertex_data = joblib.load(vertex_path)
            self.vertex_classifier = vertex_data['classifier']
            self.vertex_scaler = vertex_data['scaler']
            logger.info(f"Vertex model loaded from {vertex_path}")
        except FileNotFoundError:
            logger.warning(f"Vertex model not found at {vertex_path}")
        
        try:
            edge_data = joblib.load(edge_path)
            self.edge_classifier = edge_data['classifier']
            self.edge_scaler = edge_data['scaler']
            logger.info(f"Edge model loaded from {edge_path}")
        except FileNotFoundError:
            logger.warning(f"Edge model not found at {edge_path}")
    
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
    
    def _calculate_path_length(self, path: np.ndarray) -> float:
        """Calculate total length of a path"""
        if len(path) < 2:
            return 0.0
        
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        return np.sum(distances)
    
    def _in_bounds(self, pos: np.ndarray, shape: Tuple[int, ...]) -> bool:
        """Check if position is within bounds"""
        return all(0 <= p < s for p, s in zip(pos, shape))
    
    def _get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from vertex classifier"""
        if hasattr(self.vertex_classifier, 'feature_importances_'):
            return self.vertex_classifier.feature_importances_
        return None
    
    def _get_edge_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from edge classifier"""
        if hasattr(self.edge_classifier, 'feature_importances_'):
            return self.edge_classifier.feature_importances_
        return None

class AutomaticCurator:
    """
    Automatic curation using heuristic rules (no ML training required)
    """
    
    def __init__(self):
        pass
    
    def curate_vertices_automatic(self, vertices: Dict[str, Any], energy_data: Dict[str, Any], 
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatic vertex curation using heuristic rules
        """
        logger.info("Performing automatic vertex curation")
        
        positions = vertices['positions']
        energies = vertices['energies']
        scales = vertices['scales']
        radii = vertices['radii']
        
        # Rule-based filtering
        keep_mask = np.ones(len(positions), dtype=bool)
        
        # Rule 1: Energy threshold
        energy_threshold = parameters.get('vertex_energy_threshold', -0.1)
        keep_mask &= (energies < energy_threshold)
        
        # Rule 2: Minimum radius
        min_radius = parameters.get('min_vertex_radius', 0.5)
        keep_mask &= (radii > min_radius)
        
        # Rule 3: Distance from image boundaries
        image_shape = energy_data.get('image_shape', (100, 100, 50))
        boundary_margin = parameters.get('boundary_margin', 5)
        
        for dim in range(3):
            keep_mask &= (positions[:, dim] > boundary_margin)
            keep_mask &= (positions[:, dim] < image_shape[dim] - boundary_margin)
        
        # Rule 4: Local energy contrast
        energy_field = energy_data['energy']
        contrast_threshold = parameters.get('contrast_threshold', 0.1)
        
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
            'radii': radii[kept_indices],
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
        
        keep_mask = np.ones(len(edge_traces), dtype=bool)
        
        # Rule 1: Minimum edge length
        min_length = parameters.get('min_edge_length', 2.0)
        
        for i, trace in enumerate(edge_traces):
            if len(trace) < 2:
                keep_mask[i] = False
                continue
                
            trace = np.array(trace)
            edge_length = self._calculate_path_length(trace)
            
            if edge_length < min_length:
                keep_mask[i] = False
        
        # Rule 2: Maximum tortuosity
        max_tortuosity = parameters.get('max_edge_tortuosity', 3.0)
        
        for i, trace in enumerate(edge_traces):
            if not keep_mask[i] or len(trace) < 2:
                continue
                
            trace = np.array(trace)
            edge_length = self._calculate_path_length(trace)
            euclidean_distance = np.linalg.norm(trace[-1] - trace[0])
            
            if euclidean_distance > 0:
                tortuosity = edge_length / euclidean_distance
                if tortuosity > max_tortuosity:
                    keep_mask[i] = False
        
        # Rule 3: Valid connections
        vertex_positions = vertices['positions']
        max_connection_distance = parameters.get('max_connection_distance', 5.0)
        
        for i, (trace, connection) in enumerate(zip(edge_traces, edge_connections)):
            if not keep_mask[i]:
                continue
                
            start_vertex, end_vertex = connection
            
            # Check start connection
            if start_vertex is not None:
                start_pos = vertex_positions[start_vertex]
                trace_start = np.array(trace[0])
                if np.linalg.norm(start_pos - trace_start) > max_connection_distance:
                    keep_mask[i] = False
                    continue
            
            # Check end connection
            if end_vertex is not None:
                end_pos = vertex_positions[end_vertex]
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
    
    def _calculate_path_length(self, path: np.ndarray) -> float:
        """Calculate total length of a path"""
        if len(path) < 2:
            return 0.0
        
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        return np.sum(distances)

