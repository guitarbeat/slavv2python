#!/usr/bin/env python3
"""
Statistical Analysis for MATLAB-Python Comparison.

This script performs rigorous statistical tests on the comparison results
to quantify differences and assess their significance.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from matlab_output_parser import load_matlab_batch_results


def bootstrap_confidence_interval(data: np.ndarray, n_bootstrap: int = 10000, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI)
        
    Returns
    -------
    Tuple[float, float]
        Lower and upper confidence bounds
    """
    if len(data) == 0:
        return (0.0, 0.0)
    
    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means[i] = np.mean(sample)
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return (float(lower), float(upper))


def analyze_count_differences(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze statistical significance of count differences.
    
    Parameters
    ----------
    comparison : Dict[str, Any]
        Comparison data
        
    Returns
    -------
    Dict[str, Any]
        Statistical analysis results
    """
    results = {
        'vertices': {},
        'edges': {},
        'network': {}
    }
    
    # Vertex count analysis
    if 'vertices' in comparison and comparison['vertices']:
        verts = comparison['vertices']
        matlab_count = verts.get('matlab_count', 0)
        python_count = verts.get('python_count', 0)
        
        results['vertices'] = {
            'matlab_count': matlab_count,
            'python_count': python_count,
            'absolute_difference': abs(matlab_count - python_count),
            'relative_difference': verts.get('count_percent_difference', 0),
            'matched': verts.get('matched_vertices', 0),
            'unmatched_matlab': verts.get('unmatched_matlab', 0),
            'unmatched_python': verts.get('unmatched_python', 0)
        }
        
        # Calculate matching rate
        total_verts = max(matlab_count, python_count)
        if total_verts > 0:
            matching_rate = verts.get('matched_vertices', 0) / total_verts
            results['vertices']['matching_rate'] = float(matching_rate)
    
    # Edge count analysis
    if 'edges' in comparison and comparison['edges']:
        edges = comparison['edges']
        matlab_count = edges.get('matlab_count', 0)
        python_count = edges.get('python_count', 0)
        
        results['edges'] = {
            'matlab_count': matlab_count,
            'python_count': python_count,
            'absolute_difference': abs(matlab_count - python_count),
            'relative_difference': edges.get('count_percent_difference', 0)
        }
    
    # Network strand analysis
    if 'network' in comparison and comparison['network']:
        network = comparison['network']
        matlab_count = network.get('matlab_strand_count', 0)
        python_count = network.get('python_strand_count', 0)
        
        results['network'] = {
            'matlab_strand_count': matlab_count,
            'python_strand_count': python_count,
            'absolute_difference': abs(matlab_count - python_count),
            'relative_difference': network.get('strand_count_percent_difference', 0)
        }
    
    return results


def analyze_position_errors(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze position error statistics.
    
    Parameters
    ----------
    comparison : Dict[str, Any]
        Comparison data
        
    Returns
    -------
    Dict[str, Any]
        Position error analysis
    """
    if 'vertices' not in comparison or comparison['vertices'].get('position_rmse') is None:
        return {}
    
    verts = comparison['vertices']
    
    results = {
        'rmse': verts.get('position_rmse', 0),
        'mean_distance': verts.get('position_mean_distance', 0),
        'median_distance': verts.get('position_median_distance', 0),
        'percentile_95': verts.get('position_95th_percentile', 0),
        'interpretation': ''
    }
    
    # Interpret position errors
    rmse = results['rmse']
    if rmse < 1.0:
        results['interpretation'] = "Excellent agreement (RMSE < 1 voxel)"
    elif rmse < 2.0:
        results['interpretation'] = "Good agreement (RMSE < 2 voxels)"
    elif rmse < 5.0:
        results['interpretation'] = "Moderate agreement (RMSE < 5 voxels)"
    else:
        results['interpretation'] = f"Poor agreement (RMSE = {rmse:.2f} voxels)"
    
    return results


def analyze_radius_correlation(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze radius correlation statistics.
    
    Parameters
    ----------
    comparison : Dict[str, Any]
        Comparison data
        
    Returns
    -------
    Dict[str, Any]
        Radius correlation analysis
    """
    if 'vertices' not in comparison or not comparison['vertices'].get('radius_correlation'):
        return {}
    
    corr = comparison['vertices']['radius_correlation']
    stats_dict = comparison['vertices'].get('radius_stats', {})
    
    results = {
        'pearson': {
            'r': corr.get('pearson_r', 0),
            'p_value': corr.get('pearson_p', 1.0),
            'significant': corr.get('pearson_p', 1.0) < 0.05
        },
        'spearman': {
            'r': corr.get('spearman_r', 0),
            'p_value': corr.get('spearman_p', 1.0),
            'significant': corr.get('spearman_p', 1.0) < 0.05
        },
        'radius_stats': stats_dict,
        'interpretation': ''
    }
    
    # Interpret correlation
    pearson_r = abs(results['pearson']['r'])
    if pearson_r > 0.9:
        results['interpretation'] = "Very strong correlation (|r| > 0.9)"
    elif pearson_r > 0.7:
        results['interpretation'] = "Strong correlation (|r| > 0.7)"
    elif pearson_r > 0.5:
        results['interpretation'] = "Moderate correlation (|r| > 0.5)"
    elif pearson_r > 0.3:
        results['interpretation'] = "Weak correlation (|r| > 0.3)"
    else:
        results['interpretation'] = "Very weak or no correlation (|r| ≤ 0.3)"
    
    return results


def analyze_distribution_similarity(matlab_radii: np.ndarray, python_radii: np.ndarray) -> Dict[str, Any]:
    """Analyze distribution similarity using statistical tests.
    
    Parameters
    ----------
    matlab_radii : np.ndarray
        MATLAB radius values
    python_radii : np.ndarray
        Python radius values
        
    Returns
    -------
    Dict[str, Any]
        Distribution similarity analysis
    """
    if matlab_radii.size == 0 or python_radii.size == 0:
        return {}
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.ks_2samp(matlab_radii, python_radii)
    
    # Mann-Whitney U test (non-parametric test for difference in distributions)
    mw_stat, mw_pvalue = stats.mannwhitneyu(matlab_radii, python_radii, alternative='two-sided')
    
    # T-test for means
    t_stat, t_pvalue = stats.ttest_ind(matlab_radii, python_radii)
    
    results = {
        'kolmogorov_smirnov': {
            'statistic': float(ks_stat),
            'p_value': float(ks_pvalue),
            'significant_difference': ks_pvalue < 0.05,
            'interpretation': 'Distributions are significantly different' if ks_pvalue < 0.05 else 'No significant difference in distributions'
        },
        'mann_whitney_u': {
            'statistic': float(mw_stat),
            'p_value': float(mw_pvalue),
            'significant_difference': mw_pvalue < 0.05
        },
        't_test': {
            'statistic': float(t_stat),
            'p_value': float(t_pvalue),
            'significant_difference': t_pvalue < 0.05,
            'interpretation': 'Means are significantly different' if t_pvalue < 0.05 else 'No significant difference in means'
        }
    }
    
    return results


def compute_effect_sizes(matlab_radii: np.ndarray, python_radii: np.ndarray) -> Dict[str, float]:
    """Compute effect sizes for radius differences.
    
    Parameters
    ----------
    matlab_radii : np.ndarray
        MATLAB radius values
    python_radii : np.ndarray
        Python radius values
        
    Returns
    -------
    Dict[str, float]
        Effect size measures
    """
    if matlab_radii.size == 0 or python_radii.size == 0:
        return {}
    
    # Cohen's d (standardized mean difference)
    mean_diff = np.mean(matlab_radii) - np.mean(python_radii)
    pooled_std = np.sqrt((np.var(matlab_radii) + np.var(python_radii)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    # Hedge's g (bias-corrected version of Cohen's d)
    n1, n2 = len(matlab_radii), len(python_radii)
    correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
    hedges_g = cohens_d * correction_factor
    
    results = {
        'cohens_d': float(cohens_d),
        'hedges_g': float(hedges_g),
        'interpretation': ''
    }
    
    # Interpret effect size
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        results['interpretation'] = "Negligible effect size (|d| < 0.2)"
    elif abs_d < 0.5:
        results['interpretation'] = "Small effect size (|d| < 0.5)"
    elif abs_d < 0.8:
        results['interpretation'] = "Medium effect size (|d| < 0.8)"
    else:
        results['interpretation'] = "Large effect size (|d| ≥ 0.8)"
    
    return results


def generate_statistical_report(comparison: Dict[str, Any], matlab_data: Optional[Dict] = None, python_data: Optional[Dict] = None) -> str:
    """Generate a comprehensive statistical analysis report.
    
    Parameters
    ----------
    comparison : Dict[str, Any]
        Comparison data
    matlab_data : Optional[Dict]
        Parsed MATLAB data
    python_data : Optional[Dict]
        Python results data
        
    Returns
    -------
    str
        Formatted statistical report
    """
    report = []
    report.append("="*80)
    report.append("STATISTICAL ANALYSIS REPORT")
    report.append("MATLAB vs Python SLAVV Vectorization Comparison")
    report.append("="*80)
    report.append("")
    
    # 1. Count Differences
    report.append("1. COUNT DIFFERENCES")
    report.append("-" * 80)
    count_analysis = analyze_count_differences(comparison)
    
    if count_analysis['vertices']:
        report.append("\nVertices:")
        v = count_analysis['vertices']
        report.append(f"  MATLAB: {v['matlab_count']}")
        report.append(f"  Python: {v['python_count']}")
        report.append(f"  Absolute difference: {v['absolute_difference']}")
        report.append(f"  Relative difference: {v['relative_difference']:.2f}%")
        if 'matching_rate' in v:
            report.append(f"  Matching rate: {v['matching_rate']*100:.1f}%")
            report.append(f"  Matched vertices: {v['matched']}")
            report.append(f"  Unmatched (MATLAB only): {v['unmatched_matlab']}")
            report.append(f"  Unmatched (Python only): {v['unmatched_python']}")
    
    if count_analysis['edges']:
        report.append("\nEdges:")
        e = count_analysis['edges']
        report.append(f"  MATLAB: {e['matlab_count']}")
        report.append(f"  Python: {e['python_count']}")
        report.append(f"  Absolute difference: {e['absolute_difference']}")
        report.append(f"  Relative difference: {e['relative_difference']:.2f}%")
    
    if count_analysis['network']:
        report.append("\nNetwork Strands:")
        n = count_analysis['network']
        report.append(f"  MATLAB: {n['matlab_strand_count']}")
        report.append(f"  Python: {n['python_strand_count']}")
        report.append(f"  Absolute difference: {n['absolute_difference']}")
        report.append(f"  Relative difference: {n['relative_difference']:.2f}%")
    
    report.append("")
    
    # 2. Position Errors
    report.append("2. POSITION ERROR ANALYSIS")
    report.append("-" * 80)
    position_analysis = analyze_position_errors(comparison)
    
    if position_analysis:
        report.append(f"\n  RMSE: {position_analysis['rmse']:.3f} voxels")
        report.append(f"  Mean distance: {position_analysis['mean_distance']:.3f} voxels")
        report.append(f"  Median distance: {position_analysis['median_distance']:.3f} voxels")
        report.append(f"  95th percentile: {position_analysis['percentile_95']:.3f} voxels")
        report.append(f"\n  Interpretation: {position_analysis['interpretation']}")
    else:
        report.append("\n  Position error data not available.")
    
    report.append("")
    
    # 3. Radius Correlation
    report.append("3. RADIUS CORRELATION ANALYSIS")
    report.append("-" * 80)
    radius_corr = analyze_radius_correlation(comparison)
    
    if radius_corr:
        report.append("\nPearson Correlation:")
        report.append(f"  r = {radius_corr['pearson']['r']:.4f}")
        report.append(f"  p-value = {radius_corr['pearson']['p_value']:.4e}")
        report.append(f"  Significant: {'Yes' if radius_corr['pearson']['significant'] else 'No'} (α = 0.05)")
        
        report.append("\nSpearman Correlation:")
        report.append(f"  ρ = {radius_corr['spearman']['r']:.4f}")
        report.append(f"  p-value = {radius_corr['spearman']['p_value']:.4e}")
        report.append(f"  Significant: {'Yes' if radius_corr['spearman']['significant'] else 'No'} (α = 0.05)")
        
        if radius_corr['radius_stats']:
            rs = radius_corr['radius_stats']
            report.append("\nRadius Statistics (matched vertices):")
            report.append(f"  MATLAB: {rs.get('matlab_mean', 0):.3f} ± {rs.get('matlab_std', 0):.3f} μm")
            report.append(f"  Python: {rs.get('python_mean', 0):.3f} ± {rs.get('python_std', 0):.3f} μm")
            report.append(f"  RMSE: {rs.get('rmse', 0):.3f} μm")
        
        report.append(f"\n  Interpretation: {radius_corr['interpretation']}")
    else:
        report.append("\n  Radius correlation data not available.")
    
    report.append("")
    
    # 4. Distribution Tests (if detailed data available)
    if matlab_data and python_data:
        report.append("4. DISTRIBUTION SIMILARITY TESTS")
        report.append("-" * 80)
        
        matlab_radii = matlab_data.get('vertices', {}).get('radii', np.array([]))
        python_radii = python_data.get('vertices', {}).get('radii', np.array([]))
        
        if isinstance(matlab_radii, np.ndarray) and isinstance(python_radii, np.ndarray):
            if matlab_radii.size > 0 and python_radii.size > 0:
                dist_analysis = analyze_distribution_similarity(matlab_radii.flatten(), python_radii.flatten())
                
                if dist_analysis:
                    report.append("\nKolmogorov-Smirnov Test:")
                    ks = dist_analysis['kolmogorov_smirnov']
                    report.append(f"  Statistic: {ks['statistic']:.4f}")
                    report.append(f"  p-value: {ks['p_value']:.4e}")
                    report.append(f"  {ks['interpretation']}")
                    
                    report.append("\nT-test for Means:")
                    tt = dist_analysis['t_test']
                    report.append(f"  Statistic: {tt['statistic']:.4f}")
                    report.append(f"  p-value: {tt['p_value']:.4e}")
                    report.append(f"  {tt['interpretation']}")
                    
                    # Effect sizes
                    effect = compute_effect_sizes(matlab_radii.flatten(), python_radii.flatten())
                    report.append("\nEffect Size:")
                    report.append(f"  Cohen's d: {effect['cohens_d']:.4f}")
                    report.append(f"  Hedges' g: {effect['hedges_g']:.4f}")
                    report.append(f"  {effect['interpretation']}")
        
        report.append("")
    
    # 5. Performance Analysis
    report.append("5. PERFORMANCE COMPARISON")
    report.append("-" * 80)
    
    if 'performance' in comparison and comparison['performance']:
        perf = comparison['performance']
        report.append(f"\n  MATLAB time: {perf.get('matlab_time_seconds', 0):.2f} seconds")
        report.append(f"  Python time: {perf.get('python_time_seconds', 0):.2f} seconds")
        if 'speedup' in perf:
            report.append(f"  Speedup factor: {abs(perf['speedup']):.2f}x")
            report.append(f"  Faster implementation: {perf['faster']}")
    else:
        report.append("\n  Performance data not available.")
    
    report.append("")
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description='Perform statistical analysis on MATLAB-Python comparison'
    )
    parser.add_argument(
        '--comparison-report',
        required=True,
        help='Path to comparison_report.json'
    )
    parser.add_argument(
        '--matlab-batch',
        help='Path to MATLAB batch folder (for detailed analysis)'
    )
    parser.add_argument(
        '--python-results',
        help='Path to Python results directory'
    )
    parser.add_argument(
        '--output',
        default='comparison_output/statistical_analysis.txt',
        help='Output file for analysis report'
    )
    parser.add_argument(
        '--json-output',
        help='Optional JSON output file for structured results'
    )
    
    args = parser.parse_args()
    
    # Load comparison report
    report_path = Path(args.comparison_report)
    if not report_path.exists():
        print(f"ERROR: Comparison report not found: {report_path}")
        return 1
    
    with open(report_path, 'r') as f:
        comparison = json.load(f)
    
    # Load detailed MATLAB data if available
    matlab_data = None
    if args.matlab_batch:
        try:
            matlab_data = load_matlab_batch_results(args.matlab_batch)
        except Exception as e:
            print(f"Warning: Could not load MATLAB data: {e}")
    
    # Load Python data if available
    python_data = None
    if args.python_results:
        python_results_dir = Path(args.python_results)
        import pickle
        for pkl_file in python_results_dir.glob('*.pkl'):
            try:
                with open(pkl_file, 'rb') as f:
                    python_data = pickle.load(f)
                break
            except:
                pass
    
    # Generate report
    print("\nGenerating statistical analysis...")
    report_text = generate_statistical_report(comparison, matlab_data, python_data)
    
    # Save text report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nStatistical analysis saved to: {output_path}")
    
    # Save JSON report if requested
    if args.json_output:
        structured_results = {
            'count_differences': analyze_count_differences(comparison),
            'position_errors': analyze_position_errors(comparison),
            'radius_correlation': analyze_radius_correlation(comparison)
        }
        
        with open(args.json_output, 'w') as f:
            json.dump(structured_results, f, indent=2)
        
        print(f"Structured results saved to: {args.json_output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
