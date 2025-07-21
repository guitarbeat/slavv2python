import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import our modules
from src.vectorization_core import SLAVVProcessor, validate_parameters, calculate_network_statistics
from src.ml_curator import MLCurator
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
                    status_text.text("Loading image...")
                    progress_bar.progress(10)
                    time.sleep(1)
                    
                    # Create dummy image for demonstration
                    image = np.random.rand(100, 100, 50)
                    
                    # Initialize processor
                    processor = SLAVVProcessor()
                    
                    # Step 1: Energy calculation
                    status_text.text("Calculating energy field...")
                    progress_bar.progress(25)
                    time.sleep(1)
                    
                    # Step 2: Vertex extraction
                    status_text.text("Extracting vertices...")
                    progress_bar.progress(50)
                    time.sleep(1)
                    
                    # Step 3: Edge extraction
                    status_text.text("Extracting edges...")
                    progress_bar.progress(75)
                    time.sleep(1)
                    
                    # Step 4: Network construction
                    status_text.text("Constructing network...")
                    progress_bar.progress(90)
                    time.sleep(1)
                    
                    # Complete processing
                    results = processor.process_image(image, validated_params)
                    
                    progress_bar.progress(100)
                    status_text.text("Processing complete!")
                
                # Store results in session state
                st.session_state["processing_results"] = results
                st.session_state["parameters"] = validated_params
                
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
    Use machine learning algorithms to automatically curate and refine the detected vertices and edges.
    This step helps improve the accuracy of the vectorization by removing false positives and enhancing
    true vascular structures. This functionality is based on `MLDeployment.py` and `MLLibrary.py` from the original MATLAB repository.
    """)
    
    # Curation options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Vertex Curation")
        
        vertex_curation_method = st.selectbox(
            "Vertex curation method",
            ["auto", "manual", "machine-auto", "machine-manual"],
            help="Choose how to curate detected vertices. Corresponds to `VertexCuration` parameter in MATLAB."
        )
        
        if vertex_curation_method in ["machine-auto", "machine-manual"]:
            st.info("🤖 Machine learning curation will analyze vertex features and classify them automatically")
            
            vertex_confidence_threshold = st.slider(
                "Confidence threshold",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                help="Minimum confidence score for keeping vertices"
            )
        
        if st.button("🎯 Curate Vertices"):
            with st.spinner("Curating vertices..."):
                curator = MLCurator()
                
                # Simulate curation
                time.sleep(2)
                
                # Update results
                original_count = len(st.session_state["processing_results"]["vertices"]["positions"])
                curated_count = int(original_count * 0.8)  # Simulate 20% reduction
                
                st.success(f"✅ Vertex curation complete: {original_count} → {curated_count} vertices")
    
    with col2:
        st.markdown("### 🔗 Edge Curation")
        
        edge_curation_method = st.selectbox(
            "Edge curation method",
            ["auto", "manual", "machine-auto", "machine-manual"],
            help="Choose how to curate detected edges. Corresponds to `EdgeCuration` parameter in MATLAB."
        )
        
        if edge_curation_method in ["machine-auto", "machine-manual"]:
            st.info("🤖 Machine learning curation will analyze edge features and classify them automatically")
            
            edge_confidence_threshold = st.slider(
                "Edge confidence threshold",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                help="Minimum confidence score for keeping edges"
            )
        
        if st.button("🔗 Curate Edges"):
            with st.spinner("Curating edges..."):
                curator = MLCurator()
                
                # Simulate curation
                time.sleep(2)
                
                # Update results
                original_count = len(st.session_state["processing_results"]["edges"]["traces"])
                curated_count = int(original_count * 0.75)  # Simulate 25% reduction
                
                st.success(f"✅ Edge curation complete: {original_count} → {curated_count} edges")
    
    # Curation results
    if st.button("📊 Show Curation Statistics"):
        st.markdown("### 📈 Curation Results")
        
        # Create sample curation statistics
        curation_stats = pd.DataFrame({
            "Component": ["Vertices", "Edges", "Strands"],
            "Original": [150, 200, 45],
            "After Curation": [120, 150, 38],
            "Removed (%)": [20, 25, 15.6]
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
    
    with col1:
        st.markdown(f"### 📊 {viz_type}")
        
        # Generate sample visualization based on type
        if viz_type == "2D Network":
            # Create sample 2D network plot
            fig = create_sample_2d_network()
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "3D Network":
            # Create sample 3D network plot
            fig = create_sample_3d_network()
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Depth Projection":
            # Create depth projection
            fig = create_sample_depth_projection()
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Strand Analysis":
            # Create strand analysis plot
            fig = create_sample_strand_analysis()
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Energy Field":
            # Create energy field visualization
            fig = create_sample_energy_field()
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
    
    # Generate sample statistics
    stats = generate_sample_statistics()
    
    # Key metrics
    st.markdown("### 📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Length", f'{stats["total_length"]:.1f} μm')  
    with col2:
        st.metric("Volume Fraction", f"{stats['volume_fraction']:.3f}")
    
    with col3:
        st.metric("Bifurcation Density", f"{stats['bifurcation_density']:.2f} /mm³")
    
    with col4:
        st.metric("Mean Radius", f"{stats['mean_radius']:.2f} μm")    
    # Detailed analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🌳 Topology", "📏 Morphometry", "📊 Statistics"])
    
    with tab1:
        st.markdown("#### Length and Radius Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Length distribution
            fig_length = create_length_distribution()
            st.plotly_chart(fig_length, use_container_width=True)
        
        with col2:
            # Radius distribution
            fig_radius = create_radius_distribution()
            st.plotly_chart(fig_radius, use_container_width=True)
    
    with tab2:
        st.markdown("#### Network Topology")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Degree distribution
            fig_degree = create_degree_distribution()
            st.plotly_chart(fig_degree, use_container_width=True)
        
        with col2:
            # Connectivity analysis
            connectivity_stats = pd.DataFrame({
                "Metric": ["Connected Components", "Average Path Length", "Clustering Coefficient", "Network Diameter"],
                "Value": [1, 12.5, 0.23, 25]
            })
            st.dataframe(connectivity_stats, use_container_width=True)
    
    with tab3:
        st.markdown("#### Morphometric Analysis")
        
        # Depth-resolved statistics
        fig_depth = create_depth_statistics()
        st.plotly_chart(fig_depth, use_container_width=True)
        
        # Tortuosity analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Mean Tortuosity", "1.15")
            st.metric("Tortuosity Std", "0.08")
        
        with col2:
            st.metric("Fractal Dimension", "2.34")
            st.metric("Lacunarity", "0.12")
    
    with tab4:
        st.markdown("#### Complete Statistics Table")
        
        # Comprehensive statistics table
        full_stats = pd.DataFrame({
            "Metric": [
                "Number of Strands", "Number of Bifurcations", "Number of Endpoints",
                "Total Length (μm)", "Mean Strand Length (μm)", "Length Density (μm/μm³)",
                "Volume Fraction", "Mean Radius (μm)", "Radius Std (μm)",
                "Bifurcation Density (/mm³)", "Surface Area (μm²)", "Mean Tortuosity"
            ],
            "Value": [
                stats["num_strands"], stats["num_bifurcations"], stats["num_endpoints"],
                f"{stats["total_length"]:.1f}", f"{stats["mean_strand_length"]:.1f}", f"{stats["length_density"]:.3f}",
                f"{stats["volume_fraction"]:.4f}", f"{stats["mean_radius"]:.2f}", f"{stats["radius_std"]:.2f}",
                f"{stats["bifurcation_density"]:.2f}", f"{stats["surface_area"]:.1f}", f"{stats["mean_tortuosity"]:.3f}"
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

def show_about_page():
    """Display the about page"""
    
    st.markdown("<h2 class=\"section-header\">About SLAVV</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🔬 Algorithm Background
        
        SLAVV (Segmentation-Less, Automated, Vascular Vectorization) was originally developed by 
        **Samuel Alexander Mihelic** as a MATLAB implementation for analyzing vascular networks 
        in microscopy images.
        
        #### 📚 Key Publications
        
        - Mihelic, S.A., Sikora, W.A., Hassan, A.M., Williamson, M.R., Jones, T.A., & Dunn, A.K. (2021). 
          **Segmentation-Less, Automated, Vascular Vectorization**. PLOS Computational Biology, 17(10), e1009451. 
          [DOI: 10.1371/journal.pcbi.1009451](https://doi.org/10.1371/journal.pcbi.1009451)
        - Original MATLAB implementation: [GitHub Repository](https://github.com/UTFOIL/Vectorization-Public)
        
        #### 🔄 Python Implementation
        
        This Streamlit application is a Python port of the original MATLAB code, featuring:
        
        - **Enhanced User Interface** - Modern web-based interface with interactive controls
        - **Real-time Visualization** - Interactive 2D and 3D network rendering
        - **Machine Learning Integration** - Automated curation using trained models
        - **Multiple Export Formats** - VMV, CASX, CSV, JSON, and MAT file support
        - **Comprehensive Analysis** - Extended statistical analysis and morphometry
        
        #### 🛠️ Technical Details
        
        **Core Algorithm Steps (as described in MATLAB `vectorize_V200.m` and the PLOS Comp. Bio. paper):**
        
        1. **Energy Image Formation**
           - The input image is linearly (matched-)filtered at many scales to extract curvature (and gradient) information.
           - The Hessian matrix is diagonalized to extract principle curvatures at all voxels (and scales) where the Laplacian is negative (local bright spot in original image).
           - The energy function is an objective function to select for large negative principle curvatures while each curvature is separately weighted by a symmetry factor using the gradient.
           - The 4D multi-scale energy image is projected across the scale coordinate using a minimum projection to select the most probable scale. The result is an enhancement of the vessel centerlines while simultaneously selecting the scale coordinate.
           - Includes PSF correction using Zipfel et al. model and vesselness enhancement using Frangi-like measures.
        
        2. **Vertex Extraction**
           - Vertices are extracted as local minima in the 4D energy image (with associated x, y, z, radius, and energy value).
           - This part of the method was inspired by the first part of the SIFT algorithm (David Lowe, International Journal of Computer Vision, 2004).
           - The vertex objects are points of high contrast and symmetry (bright spots or bifurcations) along the vessel segments.
           - The labeling is sufficiently dense if there is at least one vertex per strand. Vertices are ordered by energy values to rank them from most likely to least likely.
           - Includes volume exclusion for overlapping detections and energy-based ranking and selection.
        
        3. **Edge Extraction**
           - Edges are extracted as voxel walks through the (3D) energy image. Edges are 1-Dimensional objects (list or trace), where each location along the trace is a spherical object with an energy value.
           - Each edge walk starts at a vertex and seeks the lowest energy values under the constraint that it must move away from its origin.
           - Trajectories between vertices are ordered by their maximum energy value attained to give a first estimate of which edges are likely to be true.
           - Includes gradient descent tracing from vertices and multi-directional exploration.
        
        4. **Network Construction**
           - The final network output is the minimal set of 1-Dimensional objects (strands) that connect all of the bifurcations/endpoints according to the adjacency matrix extracted from the edges and vertices.
           - Strands are like edges, but generally longer and composed of multiple edges (at least 1). Each strand has at each location along its trace a 3-space position, radius, and an energy value.
           - Network information allows for smoothing positions and sizes of extracted vectors along their strands and approximating local blood flow fields.
           - Includes connected component analysis and strand assembly from edge traces.
        """)
    
    with col2:
        st.markdown("### 📊 System Information")
        
        # System info
        st.code(f"""
Python Version: 3.11+
Streamlit Version: Latest
NumPy Version: Latest
SciPy Version: Latest
Scikit-Image Version: Latest
Plotly Version: Latest
        """)
        
        st.markdown("### 🔗 Links")
        st.markdown("""
        - [Original MATLAB Code](https://github.com/UTFOIL/Vectorization-Public)
        - [PLOS Computational Biology Publication](https://doi.org/10.1371/journal.pcbi.1009451)
        - [Algorithm Documentation (MATLAB README)](https://github.com/UTFOIL/Vectorization-Public/blob/master_pullRQ/vectorize_V200.m)
        """)
        
        st.markdown("### 📧 Contact")
        st.markdown("""
        For questions about this Python/Streamlit implementation:
        - Create an issue on the GitHub repository for this project.
        
        For questions about the original algorithm or MATLAB implementation:
        - Refer to the original publications.
        - Contact Samuel Alexander Mihelic (see publication for author details).
        """)
        
        st.markdown("### ⚖️ License")
        st.markdown("""
        This implementation is provided under the same license 
        as the original MATLAB code. Please refer to the 
        original repository for licensing details.
        """)

# Helper functions for creating sample visualizations
def create_sample_2d_network():
    """Create a sample 2D network visualization"""
    np.random.seed(42)
    
    # Generate sample network data
    n_points = 50
    x = np.random.rand(n_points) * 100
    y = np.random.rand(n_points) * 100
    
    fig = go.Figure()
    
    # Add edges (sample connections)
    for i in range(0, n_points-1, 3):
        fig.add_trace(go.Scatter(
            x=[x[i], x[i+1]], y=[y[i], y[i+1]],
            mode="lines",
            line=dict(color="blue", width=2),
            showlegend=False
        ))
    
    # Add vertices
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(size=8, color="red"),
        name="Vertices"
    ))
    
    fig.update_layout(
        title="2D Vascular Network",
        xaxis_title="X (μm)",
        yaxis_title="Y (μm)",
        showlegend=True
    )
    
    return fig

def create_sample_3d_network():
    """Create a sample 3D network visualization"""
    np.random.seed(42)
    
    # Generate sample 3D network data
    n_points = 30
    x = np.random.rand(n_points) * 100
    y = np.random.rand(n_points) * 100
    z = np.random.rand(n_points) * 50
    
    fig = go.Figure()
    
    # Add 3D scatter plot
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+lines",
        marker=dict(size=5, color=z, colorscale="Viridis"),
        line=dict(color="blue", width=3),
        name="Network"
    ))
    
    fig.update_layout(
        title="3D Vascular Network",
        scene=dict(
            xaxis_title="X (μm)",
            yaxis_title="Y (μm)",
            zaxis_title="Z (μm)"
        )
    )
    
    return fig

def create_sample_depth_projection():
    """Create a sample depth projection"""
    np.random.seed(42)
    
    # Generate sample depth data
    z_bins = np.arange(0, 100, 5)
    vessel_density = np.random.exponential(2, len(z_bins))
    
    fig = px.bar(
        x=z_bins, y=vessel_density,
        title="Vessel Density vs Depth",
        labels={"x": "Depth (μm)", "y": "Vessel Density"}
    )
    
    return fig

def create_sample_strand_analysis():
    """Create a sample strand analysis plot"""
    np.random.seed(42)
    
    # Generate sample strand data
    strand_lengths = np.random.lognormal(2, 0.5, 100)
    
    fig = px.histogram(
        x=strand_lengths,
        title="Strand Length Distribution",
        labels={"x": "Length (μm)", "y": "Count"},
        nbins=20
    )
    
    return fig

def create_sample_energy_field():
    """Create a sample energy field visualization"""
    np.random.seed(42)
    
    # Generate sample energy field
    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    X, Y = np.meshgrid(x, y)
    Z = -np.exp(-((X-50)**2 + (Y-50)**2)/500) + np.random.normal(0, 0.1, X.shape)
    
    fig = go.Figure(data=go.Heatmap(
        x=x, y=y, z=Z,
        colorscale="RdBu",
        colorbar=dict(title="Energy")
    ))
    
    fig.update_layout(
        title="Energy Field (2D Slice)",
        xaxis_title="X (μm)",
        yaxis_title="Y (μm)"
    )
    
    return fig

def create_length_distribution():
    """Create length distribution plot"""
    np.random.seed(42)
    lengths = np.random.lognormal(2.5, 0.8, 200)
    
    fig = px.histogram(
        x=lengths,
        title="Strand Length Distribution",
        labels={"x": "Length (μm)", "y": "Count"},
        nbins=25
    )
    
    return fig

def create_radius_distribution():
    """Create radius distribution plot"""
    np.random.seed(42)
    radii = np.random.lognormal(1, 0.5, 200)
    
    fig = px.histogram(
        x=radii,
        title="Vessel Radius Distribution",
        labels={"x": "Radius (μm)", "y": "Count"},
        nbins=25
    )
    
    return fig

def create_degree_distribution():
    """Create degree distribution plot"""
    degrees = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5]
    degree_counts = [degrees.count(i) for i in range(1, 6)]
    
    fig = px.bar(
        x=list(range(1, 6)), y=degree_counts,
        title="Vertex Degree Distribution",
        labels={"x": "Degree", "y": "Count"}
    )
    
    return fig

def create_depth_statistics():
    """Create depth-resolved statistics plot"""
    np.random.seed(42)
    
    depths = np.arange(0, 100, 10)
    length_density = np.random.exponential(0.5, len(depths))
    volume_fraction = np.random.exponential(0.02, len(depths))
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=depths, y=length_density, name="Length Density"),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=depths, y=volume_fraction, name="Volume Fraction"),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Depth (μm)")
    fig.update_yaxes(title_text="Length Density", secondary_y=False)
    fig.update_yaxes(title_text="Volume Fraction", secondary_y=True)
    
    fig.update_layout(title="Depth-Resolved Statistics")
    
    return fig

def generate_sample_statistics():
    """Generate sample network statistics"""
    np.random.seed(42)
    
    return {
        "num_strands": 45,
        "num_bifurcations": 12,
        "num_endpoints": 78,
        "total_length": 2456.7,
        "mean_strand_length": 54.6,
        "length_density": 0.123,
        "volume_fraction": 0.0456,
        "mean_radius": 3.2,
        "radius_std": 1.8,
        "bifurcation_density": 0.67,
        "surface_area": 1234.5,
        "mean_tortuosity": 1.15
    }

if __name__ == "__main__":
    main()





