
import sys
import os
import numpy as np
import logging

# Add project root to path to allow importing src.slavv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.slavv.vectorization_core import SLAVVProcessor
from src.slavv.io_utils import load_tiff_volume, export_pipeline_results

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("--- SLAVV Tutorial Reproduction Script ---")
    
    # 1. Ask for Input File
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        print("\nNote: The tutorial data is not included in the repository due to size.")
        print("Please download 'Image A' (20170809_txRed_chronic_tiled.tif) from the Google Drive linked in Vectorization-Public/README.md")
        input_path = input("Enter path to the .tif file (or press Enter to exit): ").strip()
        
    if not input_path:
        print("No file provided. Exiting.")
        return

    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    print(f"Loading {input_path}...")
    try:
        image = load_tiff_volume(input_path)
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    print(f"Image shape: {image.shape}, Data type: {image.dtype}")

    # 2. Define Parameters (from vectorization_script_2017MMDD_TxRed_chronic.m)
    # Original MATLAB values:
    # microns_per_voxel = [1.07, 1.07, 5]
    # radius_of_smallest_vessel_in_microns = 1.5
    # radius_of_largest_vessel_in_microns = 60
    # gaussian_to_ideal_ratio = 0.5
    # spherical_to_annular_ratio = 6/7 (~0.857)
    # approximating_PSF = true
    # excitation_wavelength = 1.3 * 2 = 2.6
    # scales_per_octave = 3
    # max_voxels_per_node_energy = 1e6

    print("Configuring parameters from tutorial...")
    params = {
        'microns_per_voxel': np.array([1.07, 1.07, 5.0]), # Anisotropic!
        'radius_of_smallest_vessel_in_microns': 1.5,
        'radius_of_largest_vessel_in_microns': 60.0,
        'gaussian_to_ideal_ratio': 0.5,
        'spherical_to_annular_ratio': 6.0/7.0,
        'approximating_PSF': True,
        'numerical_aperture': 0.95, # Default
        'excitation_wavelength_in_microns': 2.6, # 1.3 * fudge factor 2
        'scales_per_octave': 3.0,
        'energy_sign': -1.0, # Negative for bright vessels (standard)
        'max_voxels_per_node_energy': 1e9, # Disable chunking to avoid small-chunk gradient errors
        'energy_method': 'hessian', # Tutorial uses default Hessian
    }

    # 3. Run Pipeline
    print("Initializing Processor...")
    processor = SLAVVProcessor()
    print("Running Pipeline (Energy -> Vertices -> Edges -> Network)...")
    def progress_callback(frac, stage):
        print(f"Progress: {frac*100:.0f}% - {stage}")

    # Use built-in checkpointing to resume if interrupted
    print(f"Resume enabled: Checkpointing to 'tutorial_output/'")
    
    try:
        results = processor.process_image(
            image, 
            params, 
            progress_callback=progress_callback,
            checkpoint_dir="tutorial_output"
        )
            
        # 4. Results
        vertices = results['vertices']['positions']
        edges = results['edges']['traces']
        network_strands = results['network']['strands']
        
        print("\n--- Tutorial Reproduction Successful ---")
        print(f"Vertices found: {len(vertices)}")
        print(f"Edges traced: {len(edges)}")
        print(f"Network strands: {len(network_strands)}")

        # 5. Export Standard Results
        print("Exporting results...")
        export_pipeline_results(results, "tutorial_output", base_name="tutorial")
        
        # 6. Export VMV for VessMorphoVis/Blender visualization
        try:
            from src.slavv.visualization import NetworkVisualizer
            visualizer = NetworkVisualizer()
            
            # Export to VMV format for Blender visualization
            vmv_path = visualizer.export_network_data(
                results, 
                "tutorial_output/network.vmv", 
                format='vmv'
            )
            print(f"VMV export for VessMorphoVis: {vmv_path}")
            
            # Export to CASX format
            casx_path = visualizer.export_network_data(
                results, 
                "tutorial_output/network.casx", 
                format='casx'
            )
            print(f"CASX export: {casx_path}")
        except Exception as e:
            print(f"Note: Visualization export skipped: {e}")
        
        print("Done! Results saved to tutorial_output/")
        
    except Exception as e:
        print(f"\nProcessing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
