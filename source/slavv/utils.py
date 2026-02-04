
"""
Utility functions for SLAVV.
Includes preprocessing, parameter validation, and statistical helpers.
"""
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import scipy.ndimage as ndi
import warnings

def calculate_path_length(path: np.ndarray) -> float:
    """Calculate the total length of a polyline path."""
    if len(path) < 2:
        return 0.0
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return float(np.sum(distances))

def weighted_ks_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    weights1: Optional[np.ndarray] = None,
    weights2: Optional[np.ndarray] = None,
) -> float:
    """Compute the two-sample weighted Kolmogorov–Smirnov statistic."""
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)
    w1 = np.ones_like(sample1, dtype=float) if weights1 is None else np.asarray(weights1, dtype=float)
    w2 = np.ones_like(sample2, dtype=float) if weights2 is None else np.asarray(weights2, dtype=float)

    if sample1.size == 0 or sample2.size == 0:
        return 0.0

    # Sort samples and weights together
    order1 = np.argsort(sample1)
    order2 = np.argsort(sample2)
    x = sample1[order1]
    y = sample2[order2]
    w1 = w1[order1]
    w2 = w2[order2]

    cdf1 = np.cumsum(w1) / np.sum(w1)
    cdf2 = np.cumsum(w2) / np.sum(w2)

    # Evaluate CDFs at all unique points from both samples
    all_values = np.concatenate([x, y])
    sorted_values = np.sort(all_values)
    idx1 = np.searchsorted(x, sorted_values, side="right")
    idx2 = np.searchsorted(y, sorted_values, side="right")
    cdf1_vals = np.concatenate([[0.0], cdf1])[idx1]
    cdf2_vals = np.concatenate([[0.0], cdf2])[idx2]

    return float(np.max(np.abs(cdf1_vals - cdf2_vals)))

def preprocess_image(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Normalize intensities and optionally correct axial banding.

    Parameters
    ----------
    image:
        Raw input image array (y, x, z).
    params:
        Processing parameters. Uses ``bandpass_window`` to control the
        axial Gaussian smoothing window for band removal.

    Returns
    -------
    np.ndarray
        Preprocessed image with intensities scaled to ``[0, 1]``.
    """
    img = image.astype(np.float32)

    # Scale intensities to [0, 1]
    img -= img.min()
    max_val = img.max()
    if max_val > 0:
        img /= max_val

    # Simple axial band correction inspired by `fix_intensity_bands.m`
    band_window = params.get("bandpass_window", 0.0)
    if band_window > 0:
        background = ndi.gaussian_filter(img, sigma=(0, 0, band_window))
        img = np.clip(img - background, 0, 1)

    return img

def validate_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and set default parameters based on MATLAB implementation
    """
    validated = {}
    
    # Voxel size parameters
    validated['microns_per_voxel'] = params.get('microns_per_voxel', [1.0, 1.0, 1.0])
    if len(validated['microns_per_voxel']) != 3:
        raise ValueError("microns_per_voxel must be a 3-element array")
    
    # Vessel size parameters
    validated['radius_of_smallest_vessel_in_microns'] = params.get(
        'radius_of_smallest_vessel_in_microns', 1.5)
    validated['radius_of_largest_vessel_in_microns'] = params.get(
        'radius_of_largest_vessel_in_microns', 50.0)
    
    if validated['radius_of_smallest_vessel_in_microns'] <= 0:
        raise ValueError("radius_of_smallest_vessel_in_microns must be positive")
    if validated['radius_of_largest_vessel_in_microns'] <= validated['radius_of_smallest_vessel_in_microns']:
        raise ValueError("radius_of_largest_vessel_in_microns must be larger than smallest")
    
    # PSF parameters
    validated['approximating_PSF'] = params.get('approximating_PSF', True)
    if validated['approximating_PSF']:
        validated['numerical_aperture'] = params.get('numerical_aperture', 0.95)
        validated['excitation_wavelength_in_microns'] = params.get(
            'excitation_wavelength_in_microns', 1.3)
        validated['sample_index_of_refraction'] = params.get(
            'sample_index_of_refraction', 1.33)
        
        # Validate excitation wavelength (common range for two-photon microscopy)
        if not (0.7 <= validated['excitation_wavelength_in_microns'] <= 3.0):
            warnings.warn(
                "Excitation wavelength outside typical range (0.7-3.0 μm). "
                "This may indicate an error or unusual experimental setup."
            )
    
    # Scale parameters
    validated['scales_per_octave'] = params.get('scales_per_octave', 1.5)
    validated['gaussian_to_ideal_ratio'] = params.get('gaussian_to_ideal_ratio', 1.0)
    validated['spherical_to_annular_ratio'] = params.get('spherical_to_annular_ratio', 1.0)
    validated['energy_sign'] = params.get('energy_sign', -1.0)
    if validated['energy_sign'] not in (-1, 1):
        raise ValueError('energy_sign must be -1 or 1')
    if validated['scales_per_octave'] <= 0:
        raise ValueError(
            'scales_per_octave must be positive (e.g., 1.5)'
        )
    if validated['gaussian_to_ideal_ratio'] <= 0:
        raise ValueError(
            'gaussian_to_ideal_ratio must be positive; try 1.0 to disable prefiltering'
        )
    if validated['spherical_to_annular_ratio'] <= 0:
        raise ValueError(
            'spherical_to_annular_ratio must be positive; use 1.0 to skip annular subtraction'
        )
    
    # Processing parameters
    validated['max_voxels_per_node_energy'] = params.get('max_voxels_per_node_energy', 1e5)
    validated['energy_upper_bound'] = params.get('energy_upper_bound', 0.0)
    validated['space_strel_apothem'] = params.get('space_strel_apothem', 1)
    validated['length_dilation_ratio'] = params.get('length_dilation_ratio', 1.0)
    validated['number_of_edges_per_vertex'] = params.get('number_of_edges_per_vertex', 4)
    validated['step_size_per_origin_radius'] = params.get('step_size_per_origin_radius', 1.0)
    validated['max_edge_energy'] = params.get('max_edge_energy', 0.0)
    validated['min_hair_length_in_microns'] = params.get('min_hair_length_in_microns', 0.0)
    validated['bandpass_window'] = params.get('bandpass_window', 0.0)
    validated['edge_method'] = params.get('edge_method', 'tracing')
    if validated['edge_method'] not in ('tracing', 'watershed'):
        raise ValueError("edge_method must be 'tracing' or 'watershed'")
    validated['energy_method'] = params.get('energy_method', 'hessian')
    if validated['energy_method'] not in ('hessian', 'frangi', 'sato'):
        raise ValueError("energy_method must be 'hessian', 'frangi', or 'sato'")
    validated['direction_method'] = params.get('direction_method', 'hessian')
    if validated['direction_method'] not in ('hessian', 'uniform'):
        raise ValueError("direction_method must be 'hessian' or 'uniform'")
    validated['return_all_scales'] = params.get('return_all_scales', False)
    if validated['max_voxels_per_node_energy'] <= 0:
        raise ValueError(
            'max_voxels_per_node_energy must be positive; increase to process larger volumes'
        )
    if validated['length_dilation_ratio'] <= 0:
        raise ValueError('length_dilation_ratio must be positive')
    if validated['number_of_edges_per_vertex'] < 1:
        raise ValueError('number_of_edges_per_vertex must be at least 1')
    if validated['step_size_per_origin_radius'] <= 0:
        raise ValueError(
            'step_size_per_origin_radius must be positive; try 0.5 for finer tracing'
        )
    if validated['min_hair_length_in_microns'] < 0:
        raise ValueError('min_hair_length_in_microns cannot be negative')
    if validated['bandpass_window'] < 0:
        raise ValueError('bandpass_window must be non-negative; set 0 to disable')
    validated['discrete_tracing'] = params.get('discrete_tracing', False)

    return validated

def get_chunking_lattice(
    shape: Tuple[int, int, int], max_voxels: int, margin: int
) -> List[Tuple[Tuple[slice, slice, slice], Tuple[slice, slice, slice], Tuple[slice, slice, slice]]]:
    """Generate overlapping z-axis chunks to limit voxel processing.

    Parameters
    ----------
    shape:
        Image shape as ``(y, x, z)``.
    max_voxels:
        Maximum voxels allowed per chunk including margins.
    margin:
        Overlap in voxels applied on both sides of each chunk along ``z``.

    Returns
    -------
    list of tuples
        ``(chunk_slice, output_slice, inner_slice)`` where ``chunk_slice``
        indexes the padded region in the source image, ``output_slice``
        corresponds to the destination region, and ``inner_slice`` selects the
        interior region of the chunk to copy into ``output_slice``.
    """

    y, x, z = shape
    plane_voxels = y * x
    max_depth = max_voxels // plane_voxels
    if max_depth <= 0 or max_depth >= z:
        return [
            (
                (slice(0, y), slice(0, x), slice(0, z)),
                (slice(0, y), slice(0, x), slice(0, z)),
                (slice(0, y), slice(0, x), slice(0, z)),
            )
        ]

    margin = min(margin, max_depth // 2)
    core_depth = max_depth - 2 * margin
    if core_depth <= 0:
        core_depth = 1

    slices = []
    start = 0
    while start < z:
        end = min(start + core_depth, z)
        pad_before = margin if start > 0 else 0
        pad_after = margin if end < z else 0
        chunk_slice = (
            slice(0, y),
            slice(0, x),
            slice(start - pad_before, end + pad_after),
        )
        output_slice = (slice(0, y), slice(0, x), slice(start, end))
        inner_slice = (
            slice(0, y),
            slice(0, x),
            slice(pad_before, pad_before + (end - start)),
        )
        slices.append((chunk_slice, output_slice, inner_slice))
        start = end

    return slices
