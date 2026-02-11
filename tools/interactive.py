"""
Interactive widgets for SLAVV development notebooks.

This module provides the UI components (widgets) used in the Jupyter notebooks
to control the vectorization pipelines and visualize results.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import matplotlib.pyplot as plt

# Import core logic
from slavv.dev.comparison import (
    load_parameters, 
    orchestrate_comparison, 
    run_standalone_comparison
)
from slavv.dev.viz import (
    set_plot_style,
    plot_count_comparison,
    plot_radius_distributions
)

# Common styles
SECTION_HEADER_STYLE = "font-size: 14px; font-weight: bold; color: #2c3e50; margin-bottom: 5px; border-bottom: 2px solid #eaeaea; padding-bottom: 5px; width: 100%"
BOX_LAYOUT = widgets.Layout(
    border='1px solid #e0e0e0', 
    padding='15px', 
    margin='10px 0', 
    border_radius='8px',
    width='95%'
)

def create_header(text: str) -> widgets.HTML:
    """Create a styled HTML header widget."""
    return widgets.HTML(f"<div style='{SECTION_HEADER_STYLE}'>{text}</div>")

def find_available_files(project_root: Path) -> List[str]:
    """Find available TIFF files in data directories."""
    search_paths = [
        project_root / "data",
        project_root / "tests" / "data"
    ]
    available_files = []
    for p in search_paths:
        if p.exists():
            available_files.extend([str(f) for f in p.glob("*.tif")])
            available_files.extend([str(f) for f in p.glob("*.tiff")])
    return sorted(available_files)

class BaseRunnerWidget:
    """Base class for pipeline runner widgets."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd().parent
        if self.project_root.name in ['scripts', 'notebooks']: # Handle if running from scripts or notebooks dir
            self.project_root = self.project_root.parent
            
        self.available_files = find_available_files(self.project_root)
        self.default_input = self.available_files[0] if self.available_files else ""
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.output_area = widgets.Output(
            layout=widgets.Layout(
                border='1px solid #ddd',
                padding='10px',
                margin='20px 0',
                max_height='400px',
                overflow='scroll'
            )
        )
        
        self.input_widget = widgets.Combobox(
            options=self.available_files,
            value=self.default_input,
            description='Input File:',
            placeholder='Type path or select from list',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        )

    def _create_file_box(self):
        return widgets.VBox([
            create_header("File Input"),
            self.input_widget
        ], layout=BOX_LAYOUT)


class MatlabRunnerWidget(BaseRunnerWidget):
    """Widget for running MATLAB vectorization."""
    
    def __init__(self, project_root: Optional[Path] = None):
        super().__init__(project_root)
        
        # specific defaults
        self.default_output = str(self.project_root / "experiments" / f"{self.timestamp}_matlab_run")
        self.default_matlab = shutil.which('matlab') or ""
        
        # Widgets
        self.matlab_widget = widgets.Text(
            value=self.default_matlab,
            description='MATLAB Path:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        )

        self.output_widget = widgets.Text(
            value=self.default_output,
            description='Output Dir:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        )
        
        self.run_button = widgets.Button(
            description='Start MATLAB Vectorization',
            button_style='info',
            layout=widgets.Layout(width='100%', height='50px'),
            icon='play-circle', 
            style={'font_weight': 'bold', 'font_size': '16px'}
        )
        self.run_button.on_click(self.on_run_clicked)
        
    def _create_path_box(self):
        return widgets.VBox([
            create_header("System Paths"),
            self.matlab_widget,
            self.output_widget
        ], layout=BOX_LAYOUT)

    def display(self):
        """Display the widget."""
        ui = widgets.VBox([
            self._create_file_box(),
            self._create_path_box(),
            self.run_button,
            self.output_area
        ])
        display(ui)
        
    def on_run_clicked(self, b):
        input_file = self.input_widget.value
        output_dir = Path(self.output_widget.value)
        matlab_path = self.matlab_widget.value
        
        # Load params (could arguably be another widget, but keeping simple for now)
        params_file = self.project_root / "notebooks" / "comparison_params.json"
        params = load_parameters(str(params_file) if params_file.exists() else None)

        self.run_button.disabled = True
        self.run_button.description = "Running MATLAB..."
        
        with self.output_area:
            clear_output()
            print(f"Starting MATLAB run at {datetime.now().strftime('%H:%M:%S')}...")
            print("-" * 40)
            
            if not os.path.exists(input_file):
                print(f"Error: Input file not found: {input_file}")
                self._reset_button()
                return

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                status = orchestrate_comparison(
                    input_file=input_file,
                    output_dir=output_dir,
                    matlab_path=matlab_path,
                    project_root=self.project_root,
                    params=params,
                    skip_matlab=False,
                    skip_python=True
                )
                
                print("-" * 40)
                if status == 0:
                    print("\nMATLAB vectorization completed successfully!")
                else:
                    print("\nMATLAB vectorization failed. Check logs above.")

            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._reset_button()
                
    def _reset_button(self):
        self.run_button.disabled = False
        self.run_button.description = 'Start MATLAB Vectorization'


class PythonRunnerWidget(BaseRunnerWidget):
    """Widget for running Python vectorization."""
    
    def __init__(self, project_root: Optional[Path] = None):
        super().__init__(project_root)
        
        # specific defaults
        self.default_output = str(self.project_root / "experiments" / f"{self.timestamp}_python_run")
        
        # Widgets
        self.output_widget = widgets.Text(
            value=self.default_output,
            description='Output Dir:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        )
        
        self.run_button = widgets.Button(
            description='Start Python Vectorization',
            button_style='success',
            layout=widgets.Layout(width='100%', height='50px'),
            icon='play-circle', 
            style={'font_weight': 'bold', 'font_size': '16px'}
        )
        self.run_button.on_click(self.on_run_clicked)
        
    def _create_path_box(self):
        return widgets.VBox([
            create_header("System Paths"),
            self.output_widget
        ], layout=BOX_LAYOUT)

    def display(self):
        """Display the widget."""
        ui = widgets.VBox([
            self._create_file_box(),
            self._create_path_box(),
            self.run_button,
            self.output_area
        ])
        display(ui)
        
    def on_run_clicked(self, b):
        input_file = self.input_widget.value
        output_dir = Path(self.output_widget.value)
        
        params_file = self.project_root / "notebooks" / "comparison_params.json"
        params = load_parameters(str(params_file) if params_file.exists() else None)

        self.run_button.disabled = True
        self.run_button.description = "Running Python..."
        
        with self.output_area:
            clear_output()
            print(f"Starting Python run at {datetime.now().strftime('%H:%M:%S')}...")
            print("-" * 40)
            
            if not os.path.exists(input_file):
                print(f"Error: Input file not found: {input_file}")
                self._reset_button()
                return

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                status = orchestrate_comparison(
                    input_file=input_file,
                    output_dir=output_dir,
                    matlab_path="", # Not needed
                    project_root=self.project_root,
                    params=params,
                    skip_matlab=True,
                    skip_python=False
                )
                
                print("-" * 40)
                if status == 0:
                    print("\nPython vectorization completed successfully!")
                else:
                    print("\nPython vectorization failed. Check logs above.")

            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._reset_button()
                
    def _reset_button(self):
        self.run_button.disabled = False
        self.run_button.description = 'Start Python Vectorization'


class ComparisonDashboardWidget:
    """Widget for running comparison and visualization."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd().parent
        if self.project_root.name in ['scripts', 'notebooks']:
            self.project_root = self.project_root.parent
            
        self.comparisons_dir = self.project_root / "experiments"
        if not self.comparisons_dir.exists():
            self.comparisons_dir.mkdir(parents=True, exist_ok=True)
            
        # Find runs
        self.all_runs = sorted([d.name for d in self.comparisons_dir.iterdir() if d.is_dir()], reverse=True)
        
        # Heuristics
        self.matlab_candidates = [r for r in self.all_runs if 'matlab' in r.lower()] + [r for r in self.all_runs if 'matlab' not in r.lower()]
        self.python_candidates = [r for r in self.all_runs if 'python' in r.lower()] + [r for r in self.all_runs if 'python' not in r.lower()]
        
        self.default_matlab = self.matlab_candidates[0] if self.matlab_candidates else None
        self.default_python = self.python_candidates[0] if self.python_candidates else None
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.default_output_name = f"{self.timestamp}_comparison_report"
        
        # Widgets
        self.matlab_widget = widgets.Dropdown(
            options=self.matlab_candidates,
            value=self.default_matlab,
            description='MATLAB Run:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        )
        
        self.python_widget = widgets.Dropdown(
            options=self.python_candidates,
            value=self.default_python,
            description='Python Run:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        )
        
        self.output_widget = widgets.Text(
            value=self.default_output_name,
            description='Output Name:',
            placeholder='Name of the new comparison folder',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        )
        
        self.run_button = widgets.Button(
            description='Compare Results',
            button_style='primary',
            layout=widgets.Layout(width='100%', height='50px'),
            icon='balance-scale', 
            style={'font_weight': 'bold', 'font_size': '16px'}
        )
        self.run_button.on_click(self.on_run_clicked)
        
        self.output_area = widgets.Output(
            layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='20px 0', max_height='400px', overflow='scroll')
        )
        
        self.last_report_path = None

    def display(self):
        """Display the widget."""
        selection_box = widgets.VBox([
            create_header("Select Runs"),
            self.matlab_widget,
            self.python_widget,
            create_header("Output"),
            self.output_widget
        ], layout=BOX_LAYOUT)
        
        display(widgets.VBox([selection_box, self.run_button, self.output_area]))

    def on_run_clicked(self, b):
        matlab_folder = self.matlab_widget.value
        python_folder = self.python_widget.value
        output_name = self.output_widget.value
        
        if not matlab_folder or not python_folder:
            with self.output_area: print("Error: Please select both runs.")
            return
            
        matlab_path = self.comparisons_dir / matlab_folder
        python_path = self.comparisons_dir / python_folder
        output_path = self.comparisons_dir / output_name

        self.run_button.disabled = True
        self.run_button.description = "Comparing..."
        
        with self.output_area:
            clear_output()
            print(f"Starting comparison at {datetime.now().strftime('%H:%M:%S')}...")
            print("-" * 40)
            
            try:
                status = run_standalone_comparison(
                    matlab_dir=matlab_path,
                    python_dir=python_path,
                    output_dir=output_path,
                    project_root=self.project_root
                )
                
                print("-" * 40)
                if status == 0:
                    print("\nComparison report generated successfully!")
                    self.last_report_path = output_path / 'comparison_report.json'
                    print("You can now visualize results using show_comparison_plots()")
                else:
                    print("\nComparison failed. Check logs above.")

            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.run_button.disabled = False
                self.run_button.description = 'Compare Results'


def show_comparison_plots(report_path: Optional[Path] = None):
    """
    Display comparison plots from a generated report.
    
    Args:
        report_path: Path to comparison_report.json. If None, tries to find latest.
    """
    set_plot_style()
    
    if report_path is None:
        # Try to find latest report in comparisons dir
        project_root = Path.cwd().parent
        if project_root.name in ['scripts', 'notebooks']:
             project_root = project_root.parent
             
        comparisons_dir = project_root / "experiments"
        if not comparisons_dir.exists():
            print("No comparisons found.")
            return

        # recursively look for json reports
        reports = list(comparisons_dir.rglob("comparison_report.json"))
        if not reports:
             print("No comparison reports found.")
             return
             
        # Sort by modification time
        report_path = sorted(reports, key=lambda f: f.stat().st_mtime)[-1]
        print(f"Using latest report: {report_path}")
    
    if not isinstance(report_path, Path):
        report_path = Path(report_path)
        
    if not report_path.exists():
        print(f"Report file not found: {report_path}")
        return
        
    # Load data
    with open(report_path, 'r') as f:
        comparison = json.load(f)
    
    # Check for empty counts
    m_counts = comparison.get('matlab', {})
    p_counts = comparison.get('python', {})
    total_elements = (
        m_counts.get('vertices_count', 0) + 
        p_counts.get('vertices_count', 0) + 
        m_counts.get('edges_count', 0) + 
        p_counts.get('edges_count', 0)
    )
    
    if total_elements == 0:
        display(HTML("<div style='background-color: #fff3cd; color: #856404; padding: 15px; border: 1px solid #ffeeba; border-radius: 4px;'><strong>Warning:</strong> The comparison data contains zero elements. Please check that your source runs produced valid output.</div>"))
        return

    # Plot 1: Counts
    print("Component Counts Comparison:")
    plot_count_comparison(comparison)
    plt.show()
    
    # Plot 2: Radii (if data available)
    # Note: Report usually only contains summaries, not full distributions unless explicitly saved
    # If we wanted to plot distributions, we'd need to load raw result files or store histograms
    # For now, we skip detailed distribution plotting if data is missing
    print("\nRadius Distribution (Placeholder - requires raw data loading):")
    # plot_radius_distributions(...)
    print("To view radius distributions, raw data loading logic needs to be enhanced.")


# Convenience functions for notebooks
def run_matlab_interface():
    widget = MatlabRunnerWidget()
    widget.display()

def run_python_interface():
    widget = PythonRunnerWidget()
    widget.display()

def run_dashboard_interface():
    widget = ComparisonDashboardWidget()
    widget.display()
