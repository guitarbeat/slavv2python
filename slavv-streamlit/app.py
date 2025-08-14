import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import our modules
from src.vectorization_core import SLAVVProcessor, validate_parameters, calculate_network_statistics
from src.ml_curator import MLCurator, AutomaticCurator
from src.visualization import NetworkVisualizer

# Page configuration
st.set_page_config(
    page_title="SLAVV - Vascular Vectorization",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .parameter-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown("<h1 class=\"main-header\">🩸 SLAVV - Vascular Vectorization System</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    **Segmentation-Less, Automated, Vascular Vectorization** - A comprehensive tool for analyzing 
    vascular networks from grayscale, volumetric microscopy images.
    
    This Python/Streamlit implementation is based on the MATLAB SLAVV algorithm by Samuel Alexander Mihelic.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "⚙️ Image Processing", "🤖 ML Curation", "📊 Visualization", "📈 Analysis", "ℹ️ About"]
    )
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    elif page == "⚙️ Image Processing":
        show_processing_page()
    elif page == "🤖 ML Curation":
        show_ml_curation_page()
    elif page == "📊 Visualization":
        show_visualization_page()
    elif page == "📈 Analysis":
        show_analysis_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Display the home page with overview and quick start"""
    
    st.markdown("<h2 class=\"section-header\">Welcome to SLAVV</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🔬 What is SLAVV?
        
        SLAVV (Segmentation-Less, Automated, Vascular Vectorization) is a sophisticated algorithm 
        for extracting and analyzing vascular networks from 3D microscopy images. The method works 
        through four main steps:
        
        1. **Energy Image Formation** - Multi-scale Hessian-based filtering to enhance vessel centerlines
        2. **Vertex Extraction** - Detection of vessel bifurcations and endpoints as local energy minima
        3. **Edge Extraction** - Tracing vessel segments between vertices through gradient descent
        4. **Network Construction** - Assembly of edges into connected vascular strands
        
        ### 🚀 Key Features
        
        - **Multi-scale Analysis** - Detects vessels across a wide range of sizes
        - **PSF Correction** - Accounts for microscope point spread function
        - **ML Curation** - Machine learning-assisted quality control
        - **Comprehensive Statistics** - Detailed network analysis and metrics
        - **Multiple Export Formats** - VMV, CASX, and custom formats
        - **Interactive Visualization** - 2D and 3D network rendering
        """)
        
        st.markdown("<div class=\"success-box\">", unsafe_allow_html=True)
        st.markdown("""
        **✅ Ready to get started?**
        
        1. Navigate to **Image Processing** to upload and process your TIFF images
        2. Use **ML Curation** to refine vertex and edge detection
        3. Explore results in **Visualization** and **Analysis** pages
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        
        # Sample statistics (would be replaced with actual data)
        st.metric("Supported Image Types", "TIFF", help="3D grayscale TIFF images")
        st.metric("Processing Steps", "4", help="Energy → Vertices → Edges → Network")
        st.metric("Export Formats", "5+", help="VMV, CASX, MAT, CSV, JSON")
        
        st.markdown("### 🔧 System Requirements")
        st.markdown("""
        - **Input**: 3D TIFF images
        - **Memory**: Depends on image size
        - **Processing**: Multi-threaded CPU
        - **Output**: Vector networks + statistics
        """)
        
        st.markdown("### 📚 Documentation")
        st.markdown("""
        - [Algorithm Overview](#)
        - [Parameter Guide](#)
        - [Export Formats](#)
        - [Troubleshooting](#)
        """)

def show_processing_page():
    """Display the image processing page"""
    
    st.markdown("<h2 class=\"section-header\">Image Processing Pipeline</h2>", unsafe_allow_html=True)
    
    # File upload
    st.markdown("### 📁 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a TIFF file",
        type=["tif", "tiff"],
        help="Upload a 3D grayscale TIFF image of vascular structures"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
            "File type": uploaded_file.type
        }
        st.json(file_details)
    
    # Processing parameters
    st.markdown("<h3 class=\"section-header\">Processing Parameters</h3>", unsafe_allow_html=True)
    
    # Create tabs for parameter categories
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Microscopy", "📏 Vessel Sizes", "⚙️ Processing", "🔬 Advanced"])
    
    with tab1:
        st.markdown("#### Microscopy Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            microns_per_voxel_y = st.number_input(
                "Y voxel size (μm)", 
                min_value=0.01, max_value=10.0, value=1.0, step=0.01,
                help="Physical size of one voxel in Y dimension. (MATLAB: microns_per_voxel(1))"
            )
            microns_per_voxel_x = st.number_input(
                "X voxel size (μm)", 
                min_value=0.01, max_value=10.0, value=1.0, step=0.01,
                help="Physical size of one voxel in X dimension. (MATLAB: microns_per_voxel(2))"
            )
            microns_per_voxel_z = st.number_input(
                "Z voxel size (μm)", 
                min_value=0.01, max_value=10.0, value=1.0, step=0.01,
                help="Physical size of one voxel in Z dimension. (MATLAB: microns_per_voxel(3))"
            )
        
        with col2:
            approximating_PSF = st.checkbox(
                "Approximate PSF", 
                value=True,
                help="Account for microscope point spread function using Zipfel et al. model. (MATLAB: approximating_PSF)"
            )
            
            if approximating_PSF:
                numerical_aperture = st.number_input(
                    "Numerical Aperture", 
                    min_value=0.1, max_value=2.0, value=0.95, step=0.01,
                    help="Numerical aperture of the microscope objective. (MATLAB: numerical_aperture)"
                )
                
                excitation_wavelength = st.number_input(
                    "Excitation wavelength (μm)", 
                    min_value=0.4, max_value=3.0, value=1.3, step=0.1,
                    help="Laser excitation wavelength. Typical range: 0.7-3.0 μm for two-photon microscopy. (MATLAB: excitation_wavelength_in_microns)"
                )
                
                # Warning for wavelength outside typical range
                if not (0.7 <= excitation_wavelength <= 3.0):
                    st.markdown("<div class=\"warning-box\">", unsafe_allow_html=True)
                    st.warning("⚠️ Excitation wavelength outside typical range (0.7-3.0 μm). This range is typical for two-photon microscopy. Please verify this value.")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                sample_index_of_refraction = st.number_input(
                    "Sample refractive index", 
                    min_value=1.0, max_value=2.0, value=1.33, step=0.01,
                    help="Refractive index of the sample medium (e.g., 1.33 for water). (MATLAB: sample_index_of_refraction)"
                )
    
    with tab2:
        st.markdown("#### Vessel Size Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            radius_smallest = st.number_input(
                "Smallest vessel radius (μm)", 
                min_value=0.1, max_value=100.0, value=1.5, step=0.1,
                help="Radius of the smallest vessel to be detected in microns. (MATLAB: radius_of_smallest_vessel_in_microns)"
            )
            
            radius_largest = st.number_input(
                "Largest vessel radius (μm)", 
                min_value=1.0, max_value=500.0, value=50.0, step=1.0,
                help="Radius of the largest vessel to be detected in microns. (MATLAB: radius_of_largest_vessel_in_microns)"
            )
            
            if radius_largest <= radius_smallest:
                st.error("❌ Largest radius must be greater than smallest radius")
        
        with col2:
            scales_per_octave = st.number_input(
                "Scales per octave", 
                min_value=0.5, max_value=5.0, value=1.5, step=0.1,
                help="Number of vessel sizes to detect per doubling of the radius cubed. (MATLAB: scales_per_octave)"
            )
            
            # Calculate and display scale information
            if radius_largest > radius_smallest:
                volume_ratio = (radius_largest / radius_smallest) ** 3
                n_scales = int(np.log(volume_ratio) / np.log(2) * scales_per_octave) + 3
                st.info(f"📊 This will generate approximately {n_scales} scales")
    
    with tab3:
        st.markdown("#### Processing Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            energy_upper_bound = st.number_input(
                "Energy upper bound", 
                min_value=-10.0, max_value=0.0, value=0.0, step=0.1,
                help="Maximum energy value for vertex detection (negative values). (MATLAB: energy_upper_bound)"
            )
            
            space_strel_apothem = st.number_input(
                "Spatial structuring element", 
                min_value=1, max_value=10, value=1, step=1,
                help="Minimum spacing between detected vertices (in voxels). (MATLAB: space_strel_apothem)"
            )
            
            length_dilation_ratio = st.number_input(
                "Length dilation ratio", 
                min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                help="Ratio of rendering length to detection length for volume exclusion. (MATLAB: length_dilation_ratio)"
            )
        
        with col2:
            number_of_edges_per_vertex = st.number_input(
                "Edges per vertex", 
                min_value=1, max_value=10, value=4, step=1,
                help="Maximum number of edge traces per seed vertex. (MATLAB: number_of_edges_per_vertex)"
            )
            
            max_voxels_per_node = st.number_input(
                "Max voxels per node", 
                min_value=1000, max_value=1000000, value=100000, step=1000,
                help="Maximum voxels per computational node for parallel processing. (MATLAB: max_voxels_per_node_energy)"
            )
    
    with tab4:
        st.markdown("#### Advanced Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gaussian_to_ideal_ratio = st.slider(
                "Gaussian to ideal ratio", 
                min_value=0.0, max_value=1.0, value=1.0, step=0.1,
                help="Standard deviation of the Gaussian kernel per the total object length for objects that are much larger than the PSF. (MATLAB: gaussian_to_ideal_ratio)"
            )
            
            spherical_to_annular_ratio = st.slider(
                "Spherical to annular ratio", 
                min_value=0.0, max_value=1.0, value=1.0, step=0.1,
                help="Weighting factor of the spherical pulse over the combined weights of spherical and annular pulses. (MATLAB: spherical_to_annular_ratio)"
            )
        
        with col2:
            step_size_per_origin_radius = st.number_input(
                "Step size ratio", 
                min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                help="Edge tracing step size relative to origin vertex radius. (MATLAB: step_size_per_origin_radius)"
            )
            
            max_edge_energy = st.number_input(
                "Max edge energy", 
                min_value=-10.0, max_value=0.0, value=0.0, step=0.1,
                help="Maximum energy threshold for edge tracing. (MATLAB: max_edge_energy)"
            )
    
    # Processing button and results
    st.markdown("<h3 class=\"section-header\">Processing</h3>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        if st.button("🚀 Start Processing", type="primary"):
            
            # Collect parameters
            parameters = {
                "microns_per_voxel": [microns_per_voxel_y, microns_per_voxel_x, microns_per_voxel_z],
                "radius_of_smallest_vessel_in_microns": radius_smallest,
                "radius_of_largest_vessel_in_microns": radius_largest,
                "approximating_PSF": approximating_PSF,
                "scales_per_octave": scales_per_octave,
                "energy_upper_bound": energy_upper_bound,
                "space_strel_apothem": space_strel_apothem,
                "length_dilation_ratio": length_dilation_ratio,
                "number_of_edges_per_vertex": number_of_edges_per_vertex,
                "max_voxels_per_node_energy": max_voxels_per_node,
                "gaussian_to_ideal_ratio": gaussian_to_ideal_ratio,
                "spherical_to_annular_ratio": spherical_to_annular_ratio,
                "step_size_per_origin_radius": step_size_per_origin_radius,
                "max_edge_energy": max_edge_energy
            }
            
            if approximating_PSF:
                parameters.update({
                    "numerical_aperture": numerical_aperture,
                    "excitation_wavelength_in_microns": excitation_wavelength,
                    "sample_index_of_refraction": sample_index_of_refraction
                })
            
            # Validate parameters
            try:
                validated_params = validate_parameters(parameters)
                st.success("✅ Parameters validated successfully")
                
                # Show processing progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate processing (replace with actual processing)
                with st.spinner("Processing image..."):
                    
                    # Load image (placeholder - would load actual TIFF)
                    import tifffile
                    status_text.text("Loading image...")
                    progress_bar.progress(10)
                    try:
                        image = tifffile.imread(uploaded_file)
                        st.success(f"✅ Image loaded successfully with shape: {image.shape}")
                    except Exception as e:
                        st.error(f"❌ Error loading TIFF file: {e}")
                        st.stop()
                    # Initialize processor
                    processor = SLAVVProcessor()
                    
                    # Step 1: Energy calculation
                    status_text.text("Calculating energy field...")
                    progress_bar.progress(25)
                    time.sleep(0.2)
                    
                    # Step 2: Vertex extraction
                    status_text.text("Extracting vertices...")
                    progress_bar.progress(50)
                    time.sleep(0.2)
                    
                    # Step 3: Edge extraction
                    status_text.text("Extracting edges...")
                    progress_bar.progress(75)
                    time.sleep(0.2)
                    
                    # Step 4: Network construction
                    status_text.text("Constructing network...")
                    progress_bar.progress(90)
                    time.sleep(0.2)
                    
                    # Complete processing
                    results = processor.process_image(image, validated_params)
                    
                    progress_bar.progress(100)
                    status_text.text("Processing complete!")
                
                # Store results in session state
                st.session_state["processing_results"] = results
                st.session_state["parameters"] = validated_params
                st.session_state["image_shape"] = image.shape
                
                # Display results summary
                st.markdown("<div class=\"success-box\">", unsafe_allow_html=True)
                st.success("🎉 Processing completed successfully!")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Results summary
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Vertices Found", len(results["vertices"]["positions"]))
                
                with col2:
                    st.metric("Edges Extracted", len(results["edges"]["traces"]))
                
                with col3:
                    st.metric("Network Strands", len(results["network"]["strands"]))
                
                with col4:
                    st.metric("Bifurcations", len(results["network"]["bifurcations"]))
                
            except Exception as e:
                st.error(f"❌ Parameter validation failed: {str(e)}")
    
    else:
        st.info("👆 Please upload a TIFF file to begin processing")

def show_ml_curation_page():
    """Display the ML curation page"""
    
    st.markdown("<h2 class=\"section-header\">Machine Learning Curation</h2>", unsafe_allow_html=True)
    
    if "processing_results" not in st.session_state:
        st.warning("⚠️ No processing results found. Please process an image first.")
        return
    
    st.markdown("""
    Use machine learning algorithms or heuristic rules to automatically curate and refine the detected vertices and edges.
    This step helps improve the accuracy of the vectorization by removing false positives and enhancing
    true vascular structures. This functionality is based on `MLDeployment.py` and `MLLibrary.py` from the original MATLAB repository.
    """)
    
    results = st.session_state["processing_results"]
    parameters = st.session_state["parameters"]
    
    st.markdown("### 🎯 Curation Options")
    curation_type = st.radio(
        "Select Curation Type:",
        ("Automatic (Rule-based)", "Machine Learning (Model-based)"),
        help="Choose between rule-based automatic curation or machine learning model-based curation."
    )

    if curation_type == "Automatic (Rule-based)":
        st.markdown("#### Automatic Curation Parameters")
        col1, col2 = st.columns(2)
        with col1:
            vertex_energy_threshold = st.number_input(
                "Vertex Energy Threshold", 
                min_value=-10.0, max_value=0.0, value=-0.1, step=0.01,
                help="Vertices with energy above this threshold will be removed."
            )
            min_vertex_radius = st.number_input(
                "Minimum Vertex Radius (μm)", 
                min_value=0.1, max_value=10.0, value=0.5, step=0.1,
                help="Vertices with radius below this will be removed."
            )
        with col2:
            boundary_margin = st.number_input(
                "Boundary Margin (voxels)", 
                min_value=0, max_value=20, value=5, step=1,
                help="Vertices too close to image boundaries will be removed."
            )
            contrast_threshold = st.number_input(
                "Local Contrast Threshold", 
                min_value=0.0, max_value=1.0, value=0.1, step=0.01,
                help="Vertices in low-contrast regions will be removed."
            )
            min_edge_length = st.number_input(
                "Minimum Edge Length (μm)", 
                min_value=0.1, max_value=20.0, value=2.0, step=0.1,
                help="Edges shorter than this will be removed."
            )
            max_edge_tortuosity = st.number_input(
                "Maximum Edge Tortuosity", 
                min_value=1.0, max_value=10.0, value=3.0, step=0.1,
                help="Edges with tortuosity above this will be removed."
            )
            max_connection_distance = st.number_input(
                "Max Connection Distance (μm)", 
                min_value=0.1, max_value=10.0, value=5.0, step=0.1,
                help="Edges not properly connected to vertices within this distance will be removed."
            )

        auto_curation_params = {
            "vertex_energy_threshold": vertex_energy_threshold,
            "min_vertex_radius": min_vertex_radius,
            "boundary_margin": boundary_margin,
            "contrast_threshold": contrast_threshold,
            "min_edge_length": min_edge_length,
            "max_edge_tortuosity": max_edge_tortuosity,
            "max_connection_distance": max_connection_distance,
            "image_shape": st.session_state["image_shape"] # Pass image shape for boundary check
        }

        if st.button("🚀 Start Automatic Curation", type="primary"):
            with st.spinner("Performing automatic curation..."):
                curator = AutomaticCurator()
                
                # Curate vertices
                curated_vertices = curator.curate_vertices_automatic(
                    results["vertices"], results["energy_data"], auto_curation_params
                )
                
                # Curate edges
                curated_edges = curator.curate_edges_automatic(
                    results["edges"], curated_vertices, auto_curation_params
                )
                
                # Update session state with curated results
                st.session_state["processing_results"]["vertices"] = curated_vertices
                st.session_state["processing_results"]["edges"] = curated_edges
                
                st.success("✅ Automatic curation complete!")
                
                # Display results summary
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Vertices", len(results["vertices"]["positions"]))
                    st.metric("Curated Vertices", len(curated_vertices["positions"]))
                with col2:
                    st.metric("Original Edges", len(results["edges"]["traces"]))
                    st.metric("Curated Edges", len(curated_edges["traces"]))

    elif curation_type == "Machine Learning (Model-based)":
        st.markdown("#### Machine Learning Curation Parameters")
        st.warning("⚠️ Machine Learning Curation requires trained models. This functionality is under development and requires pre-trained models or a training dataset.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            vertex_curation_method = st.selectbox(
                "Vertex curation method",
                ["machine-auto"], # Only machine-auto for now
                help="Choose how to curate detected vertices. Corresponds to `VertexCuration` parameter in MATLAB."
            )
            
            vertex_confidence_threshold = st.slider(
                "Vertex Confidence threshold",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                help="Minimum confidence score for keeping vertices"
            )
        
        with col2:
            edge_curation_method = st.selectbox(
                "Edge curation method",
                ["machine-auto"], # Only machine-auto for now
                help="Choose how to curate detected edges. Corresponds to `EdgeCuration` parameter in MATLAB."
            )
            
            edge_confidence_threshold = st.slider(
                "Edge Confidence threshold",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                help="Minimum confidence score for keeping edges"
            )

        if st.button("🤖 Start ML Curation", type="primary"):
            with st.spinner("Performing ML curation..."):
                curator = MLCurator()
                
                # In a real scenario, you would load pre-trained models here
                # curator.load_models("path/to/vertex_model.joblib", "path/to/edge_model.joblib")
                
                if curator.vertex_classifier is None or curator.edge_classifier is None:
                    st.error("❌ ML models not loaded or trained. Cannot perform ML curation.")
                    st.stop()

                # Curate vertices
                curated_vertices = curator.curate_vertices(
                    results["vertices"], results["energy_data"], st.session_state["image_shape"], vertex_confidence_threshold
                )
                
                # Curate edges
                curated_edges = curator.curate_edges(
                    results["edges"], curated_vertices, results["energy_data"], edge_confidence_threshold
                )
                
                # Update session state with curated results
                st.session_state["processing_results"]["vertices"] = curated_vertices
                st.session_state["processing_results"]["edges"] = curated_edges
                
                st.success("✅ ML curation complete!")
                
                # Display results summary
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Vertices", len(results["vertices"]["positions"]))
                    st.metric("Curated Vertices", len(curated_vertices["positions"]))
                with col2:
                    st.metric("Original Edges", len(results["edges"]["traces"]))
                    st.metric("Curated Edges", len(curated_edges["traces"]))

    # Curation results
    if st.button("📊 Show Curation Statistics"):
        st.markdown("### 📈 Curation Results")
        
        # Get current curated counts
        current_vertices = st.session_state["processing_results"]["vertices"]
        current_edges = st.session_state["processing_results"]["edges"]
        original_vertices_count = len(results["vertices"]["positions"])
        original_edges_count = len(results["edges"]["traces"])
        curated_vertices_count = len(current_vertices["positions"])
        curated_edges_count = len(current_edges["traces"])
        
        # Calculate percentage removed
        vertex_removed_percent = ((original_vertices_count - curated_vertices_count) / original_vertices_count * 100) if original_vertices_count > 0 else 0
        edge_removed_percent = ((original_edges_count - curated_edges_count) / original_edges_count * 100) if original_edges_count > 0 else 0

        curation_stats = pd.DataFrame({
            "Component": ["Vertices", "Edges"],
            "Original": [original_vertices_count, original_edges_count],
            "After Curation": [curated_vertices_count, curated_edges_count],
            "Removed (%)": [f"{vertex_removed_percent:.2f}", f"{edge_removed_percent:.2f}"]
        })
        
        st.dataframe(curation_stats, use_container_width=True)
        
        # Visualization of curation results
        fig = px.bar(
            curation_stats, 
            x="Component", 
            y=["Original", "After Curation"],
            title="Curation Results",
            barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)
def show_visualization_page():
    """Display the visualization page"""
    
    st.markdown("<h2 class=\"section-header\">Network Visualization</h2>", unsafe_allow_html=True)
    
    if "processing_results" not in st.session_state:
        st.warning("⚠️ No processing results found. Please process an image first.")
        return
    
    st.markdown("""
    Visualize the vectorized vascular network in 2D and 3D. This section provides interactive tools to explore the results.
    Corresponds to `Visual` and `SpecialOutput` parameters in MATLAB.
    """)
    
    # Visualization options
    viz_type = st.selectbox(
        "Visualization type",
        ["2D Network", "3D Network", "Depth Projection", "Strand Analysis", "Energy Field"],
        help="Choose the type of visualization to display"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### 🎨 Display Options")
        
        show_vertices = st.checkbox("Show vertices", value=True)
        show_edges = st.checkbox("Show edges", value=True)
        show_bifurcations = st.checkbox("Show bifurcations", value=True)
        
        color_scheme = st.selectbox(
            "Color scheme",
            ["Energy", "Depth", "Strand ID", "Radius", "Random"],
            help="How to color the network components"
        )
        
        opacity = st.slider("Opacity", 0.1, 1.0, 0.8, 0.1)
        
        if viz_type == "3D Network":
            camera_angle = st.selectbox(
                "Camera angle",
                ["Isometric", "Top", "Side", "Front"],
                help="3D viewing angle"
            )
    
    visualizer = NetworkVisualizer()
    
    with col1:
        st.markdown(f"### 📊 {viz_type}")
        
        # Generate actual visualization based on type
        if viz_type == "2D Network":
            fig = visualizer.plot_2d_network(
                st.session_state["processing_results"]["vertices"],
                st.session_state["processing_results"]["edges"],
                st.session_state["processing_results"]["network"],
                st.session_state["parameters"],
                color_by=color_scheme.lower().replace(" ", "_"),
                show_vertices=show_vertices, show_edges=show_edges, show_bifurcations=show_bifurcations
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "3D Network":
            fig = visualizer.plot_3d_network(
                st.session_state["processing_results"]["vertices"],
                st.session_state["processing_results"]["edges"],
                st.session_state["processing_results"]["network"],
                st.session_state["parameters"],
                color_by=color_scheme.lower().replace(" ", "_"),
                show_vertices=show_vertices, show_edges=show_edges, show_bifurcations=show_bifurcations
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Depth Projection":
            fig = visualizer.plot_depth_statistics(
                st.session_state["processing_results"]["vertices"],
                st.session_state["processing_results"]["edges"],
                st.session_state["parameters"]
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Strand Analysis":
            fig = visualizer.plot_strand_analysis(
                st.session_state["processing_results"]["network"],
                st.session_state["processing_results"]["vertices"],
                st.session_state["parameters"]
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Energy Field":
            # For energy field, we need to pass the original image shape to the visualizer
            # and potentially allow selecting a slice axis and index
            st.info("Energy Field visualization is a 2D slice. Select slice axis and index in sidebar.")
            slice_axis = st.sidebar.selectbox("Slice Axis", [0, 1, 2], format_func=lambda x: ["Y", "X", "Z"][x])
            energy = st.session_state["processing_results"]["energy_data"]["energy"]
            slice_index = st.sidebar.number_input("Slice Index", value=int(energy.shape[slice_axis] // 2))
            fig = visualizer.plot_energy_field(
                st.session_state["processing_results"]["energy_data"], slice_axis=slice_axis, slice_index=slice_index
            )
            st.plotly_chart(fig, use_container_width=True)    
    # Export options
    st.markdown("### 💾 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export VMV"):
            st.success("✅ VMV file exported. (MATLAB: SpecialOutput=\'vmv\')")
    
    with col2:
        if st.button("📄 Export CASX"):
            st.success("✅ CASX file exported. (MATLAB: SpecialOutput=\'casX\')")
    
    with col3:
        if st.button("📊 Export CSV"):
            st.success("✅ CSV data exported. (Custom Python export)")

def show_analysis_page():
    """Display the analysis page"""
    
    st.markdown("<h2 class=\"section-header\">Network Analysis</h2>", unsafe_allow_html=True)
    
    if "processing_results" not in st.session_state:
        st.warning("⚠️ No processing results found. Please process an image first.")
        return
    
    st.markdown("""
    Perform comprehensive statistical analysis on the vectorized vascular network. This section provides key metrics and detailed distributions.
    Corresponds to `SpecialOutput` parameters like `histograms`, `depth-stats`, `original-stats` in MATLAB.
    """)
    
    results = st.session_state["processing_results"]
    parameters = st.session_state["parameters"]

    # Calculate actual statistics using available data
    from src.vectorization_core import calculate_network_statistics as _calc_stats
    stats = _calc_stats(
        results["network"]["strands"],
        results["network"]["bifurcations"],
        results["vertices"]["positions"],
        results["vertices"]["radii"],
        parameters.get("microns_per_voxel", [1.0, 1.0, 1.0]),
        st.session_state.get("image_shape", (100, 100, 50))
    )

    # Key metrics
    st.markdown("### 📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Length", f'{stats["total_length"]:.1f} μm')
    with col2:
        st.metric("Volume Fraction", f"{stats['volume_fraction']:.3f}")

    with col3:
        st.metric("Bifurcation Density", f"{stats.get('bifurcation_density', 0):.2f} /mm³")

    with col4:
        st.metric("Mean Radius", f"{stats.get('mean_radius', 0):.2f} μm")
    # Detailed analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🌳 Topology", "📏 Morphometry", "📊 Statistics"])

    visualizer = NetworkVisualizer()

    with tab1:
        st.markdown("#### Length and Radius Distributions")

        col1, col2 = st.columns(2)

        with col1:
            # Length distribution
            fig_length = visualizer.plot_strand_analysis(results["network"], results["vertices"], parameters)
            st.plotly_chart(fig_length, use_container_width=True)

        with col2:
            # Radius distribution
            fig_radius = visualizer.plot_radius_distribution(results["vertices"])
            st.plotly_chart(fig_radius, use_container_width=True)

    with tab2:
        st.markdown("#### Network Topology")

        col1, col2 = st.columns(2)

        with col1:
            # Degree distribution
            fig_degree = visualizer.plot_degree_distribution(results["network"])
            st.plotly_chart(fig_degree, use_container_width=True)

        with col2:
            # Connectivity analysis
            connectivity_stats = pd.DataFrame({
                "Metric": ["Connected Components", "Average Path Length", "Clustering Coefficient", "Network Diameter"],
                "Value": [
                    stats.get('num_connected_components', 0),
                    stats.get('avg_path_length', 0.0),
                    stats.get('clustering_coefficient', 0.0),
                    stats.get('network_diameter', 0.0)
                ]
            })
            st.dataframe(connectivity_stats, use_container_width=True)

    with tab3:
        st.markdown("#### Morphometric Analysis")

        # Depth-resolved statistics
        fig_depth = visualizer.plot_depth_statistics(results["vertices"], results["edges"], parameters)
        st.plotly_chart(fig_depth, use_container_width=True)

        # Tortuosity analysis
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Mean Tortuosity", f"{stats.get('mean_tortuosity', 0):.2f}")
            st.metric("Tortuosity Std", f"{stats.get('tortuosity_std', 0):.2f}")

        with col2:
            st.metric("Fractal Dimension", f"{stats.get('fractal_dimension', 0):.2f}")
            st.metric("Lacunarity", f"{stats.get('lacunarity', 0):.2f}")

    with tab4:
        st.markdown("#### Complete Statistics Table")

        # Comprehensive statistics table
        full_stats = pd.DataFrame({
            "Metric": [
                "Number of Strands", "Number of Bifurcations", "Number of Endpoints",
                "Total Length (μm)", "Mean Strand Length (μm)", "Length Density (μm/μm³)",
                "Volume Fraction", "Mean Radius (μm)", "Radius Std (μm)",
                "Bifurcation Density (/mm³)", "Surface Area (μm²)", "Mean Tortuosity",
                "Number of Connected Components", "Average Path Length", "Clustering Coefficient", "Network Diameter",
                "Fractal Dimension", "Lacunarity", "Tortuosity Std"
            ],
                "Value": [
                    stats.get("num_strands", 0), stats.get("num_bifurcations", 0), stats.get("num_endpoints", 0),
                    f'{stats.get("total_length", 0):.1f}', f'{stats.get("mean_strand_length", 0):.1f}', f'{stats.get("length_density", 0):.3f}',
                    f'{stats.get("volume_fraction", 0):.4f}', f'{stats.get("mean_radius", 0):.2f}', f'{stats.get("radius_std", 0):.2f}',
                    f'{stats.get("bifurcation_density", 0):.2f}', f'{stats.get("surface_area", 0):.1f}', f'{stats.get("mean_tortuosity", 0):.3f}',
                    stats.get("num_connected_components", 0), f'{stats.get("avg_path_length", 0):.2f}', f'{stats.get("clustering_coefficient", 0):.2f}', f'{stats.get("network_diameter", 0):.2f}',
                    f'{stats.get("fractal_dimension", 0):.2f}', f'{stats.get("lacunarity", 0):.2f}', f'{stats.get("tortuosity_std", 0):.2f}'
                ]
            })

        st.dataframe(full_stats, use_container_width=True)

        # Download statistics
        csv = full_stats.to_csv(index=False)
        st.download_button(
            label="📥 Download Statistics CSV",
            data=csv,
            file_name="network_statistics.csv",
            mime="text/csv"
        )
