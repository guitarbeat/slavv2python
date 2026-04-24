"""Image processing page for the SLAVV Streamlit app."""

from __future__ import annotations

import numpy as np
import streamlit as st
from slavv.apps.processing_state import (
    load_processing_snapshot,
    store_processing_session_state,
    summarize_processing_metrics,
)
from slavv.core import SLAVVProcessor
from slavv.utils import validate_parameters

from . import app_services


def show_processing_page() -> None:
    """Display the image processing page."""
    st.markdown('<h2 class="section-header">Image Processing Pipeline</h2>', unsafe_allow_html=True)

    st.markdown("### 📁 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a TIFF file",
        type=["tif", "tiff"],
        help="Upload a 3D grayscale TIFF image of vascular structures",
    )

    if uploaded_file is not None:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        st.json(
            {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
                "File type": uploaded_file.type,
            }
        )

    st.markdown('<h3 class="section-header">Processing Parameters</h3>', unsafe_allow_html=True)
    with st.popover("Parameter tips", width=300):
        st.write(
            "Use the tabs below to adjust microscopy, vessel size, processing, "
            "and advanced options. Defaults are provided for typical datasets."
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔬 Microscopy", "📏 Vessel Sizes", "⚙️ Processing", "🔬 Advanced"]
    )

    with tab1:
        st.markdown("#### Microscopy Parameters")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            microns_per_voxel_y = st.number_input(
                "Y voxel size (μm)",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.01,
                help="Physical size of one voxel in Y dimension. (MATLAB: microns_per_voxel(1))",
            )
            microns_per_voxel_x = st.number_input(
                "X voxel size (μm)",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.01,
                help="Physical size of one voxel in X dimension. (MATLAB: microns_per_voxel(2))",
            )
            microns_per_voxel_z = st.number_input(
                "Z voxel size (μm)",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.01,
                help="Physical size of one voxel in Z dimension. (MATLAB: microns_per_voxel(3))",
            )
        with col2:
            approximating_PSF = st.checkbox(
                "Approximate PSF",
                value=True,
                help="Account for microscope point spread function using Zipfel et al. model. (MATLAB: approximating_PSF)",
            )
            if approximating_PSF:
                numerical_aperture = st.number_input(
                    "Numerical Aperture",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.95,
                    step=0.01,
                    help="Numerical aperture of the microscope objective. (MATLAB: numerical_aperture)",
                )
                excitation_wavelength = st.number_input(
                    "Excitation wavelength (μm)",
                    min_value=0.4,
                    max_value=3.0,
                    value=1.3,
                    step=0.1,
                    help="Laser excitation wavelength. Typical range: 0.7-3.0 μm for two-photon microscopy. (MATLAB: excitation_wavelength_in_microns)",
                )
                if not (0.7 <= excitation_wavelength <= 3.0):
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.warning(
                        "⚠️ Excitation wavelength outside typical range (0.7-3.0 μm). This range is typical for two-photon microscopy. Please verify this value."
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                sample_index_of_refraction = st.number_input(
                    "Sample refractive index",
                    min_value=1.0,
                    max_value=2.0,
                    value=1.33,
                    step=0.01,
                    help="Refractive index of the sample medium (e.g., 1.33 for water). (MATLAB: sample_index_of_refraction)",
                )

    with tab2:
        st.markdown("#### Vessel Size Parameters")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            radius_smallest = st.number_input(
                "Smallest vessel radius (μm)",
                min_value=0.1,
                max_value=100.0,
                value=1.5,
                step=0.1,
                help="Radius of the smallest vessel to be detected in microns. (MATLAB: radius_of_smallest_vessel_in_microns)",
            )
            radius_largest = st.number_input(
                "Largest vessel radius (μm)",
                min_value=1.0,
                max_value=500.0,
                value=50.0,
                step=1.0,
                help="Radius of the largest vessel to be detected in microns. (MATLAB: radius_of_largest_vessel_in_microns)",
            )
            if radius_largest <= radius_smallest:
                st.error("❌ Largest radius must be greater than smallest radius")
        with col2:
            scales_per_octave = st.number_input(
                "Scales per octave",
                min_value=0.5,
                max_value=5.0,
                value=1.5,
                step=0.1,
                help="Number of vessel sizes to detect per doubling of the radius cubed. (MATLAB: scales_per_octave)",
            )
            if radius_largest > radius_smallest:
                volume_ratio = (radius_largest / radius_smallest) ** 3
                n_scales = int(np.log(volume_ratio) / np.log(2) * scales_per_octave) + 3
                st.info(f"📊 This will generate approximately {n_scales} scales")

    with tab3:
        st.markdown("#### Processing Parameters")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            energy_upper_bound = st.number_input(
                "Energy upper bound",
                min_value=-10.0,
                max_value=0.0,
                value=0.0,
                step=0.1,
                help="Maximum energy value for vertex detection (negative values). (MATLAB: energy_upper_bound)",
            )
            space_strel_apothem = st.number_input(
                "Spatial structuring element",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                help="Minimum spacing between detected vertices (in voxels). (MATLAB: space_strel_apothem)",
            )
            length_dilation_ratio = st.number_input(
                "Length dilation ratio",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Ratio of rendering length to detection length for volume exclusion. (MATLAB: length_dilation_ratio)",
            )
        with col2:
            number_of_edges_per_vertex = st.number_input(
                "Edges per vertex",
                min_value=1,
                max_value=10,
                value=4,
                step=1,
                help="Maximum number of edge traces per seed vertex. (MATLAB: number_of_edges_per_vertex)",
            )
            max_voxels_per_node = st.number_input(
                "Max voxels per node",
                min_value=1000,
                max_value=1000000,
                value=100000,
                step=1000,
                help="Maximum voxels per computational node for parallel processing. (MATLAB: max_voxels_per_node_energy)",
            )

    with tab4:
        st.markdown("#### Advanced Parameters")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            gaussian_to_ideal_ratio = st.slider(
                "Gaussian to ideal ratio",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
                help="Standard deviation of the Gaussian kernel per the total object length for objects that are much larger than the PSF. (MATLAB: gaussian_to_ideal_ratio)",
            )
            spherical_to_annular_ratio = st.slider(
                "Spherical to annular ratio",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
                help="Weighting factor of the spherical pulse over the combined weights of spherical and annular pulses. (MATLAB: spherical_to_annular_ratio)",
            )
        with col2:
            step_size_per_origin_radius = st.number_input(
                "Step size ratio",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Edge tracing step size relative to origin vertex radius. (MATLAB: step_size_per_origin_radius)",
            )
            max_edge_energy = st.number_input(
                "Max edge energy",
                min_value=-10.0,
                max_value=0.0,
                value=0.0,
                step=0.1,
                help="Maximum energy threshold for edge tracing. (MATLAB: max_edge_energy)",
            )

    st.markdown('<h3 class="section-header">Processing</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        stop_after_options = {
            "Energy Field Only": "energy",
            "Energy + Vertices": "vertices",
            "Energy + Vertices + Edges": "edges",
            "Full Pipeline (Network)": "network",
        }
        stop_after_selection = st.selectbox(
            "Pipeline Target",
            options=list(stop_after_options.keys()),
            index=3,
            help="Stop the pipeline early after completing this stage. Useful for tweaking parameters.",
        )
        stop_after_val = stop_after_options[stop_after_selection]
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        force_rerun_stage = st.selectbox(
            "Force Recalculation From:",
            options=["None", "energy", "vertices", "edges", "network"],
            index=0,
            help="Ignore cached results and recalculate from this stage onwards. Leave as 'None' to use cached files if available.",
        )

    current_snapshot = load_processing_snapshot(
        st.session_state,
        snapshot_loader=app_services.load_run_snapshot,
    )
    if current_snapshot is not None:
        app_services._render_run_dashboard(current_snapshot)

    if uploaded_file is not None:
        if st.button("🚀 Start Processing", type="primary", width=250):
            parameters = {
                "microns_per_voxel": [
                    microns_per_voxel_y,
                    microns_per_voxel_x,
                    microns_per_voxel_z,
                ],
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
                "max_edge_energy": max_edge_energy,
            }
            if approximating_PSF:
                parameters.update(
                    {
                        "numerical_aperture": numerical_aperture,
                        "excitation_wavelength_in_microns": excitation_wavelength,
                        "sample_index_of_refraction": sample_index_of_refraction,
                    }
                )
            try:
                validated_params = validate_parameters(parameters)
                st.success("✅ Parameters validated successfully")

                with st.status("Processing image...", expanded=True) as status:
                    status.update(label="Loading image...", state="running")
                    try:
                        image = app_services.cached_load_tiff_volume(uploaded_file)
                        st.success(f"✅ Image loaded successfully with shape: {image.shape}")
                    except ValueError as exc:
                        st.error(f"❌ Error loading TIFF file: {exc}")
                        st.stop()

                    processor = SLAVVProcessor()
                    dashboard_placeholder = st.empty()
                    run_dir = app_services._build_processing_run_dir(
                        uploaded_file.getvalue(),
                        validated_params,
                    )

                    def event_cb(event) -> None:
                        state = "complete" if event.status.startswith("completed") else "running"
                        label = event.detail or f"{event.stage} {int(event.stage_progress * 100)}%"
                        status.update(label=label, state=state)
                        with dashboard_placeholder.container():
                            app_services._render_run_dashboard(event.snapshot)

                    results = processor.process_image(
                        image,
                        validated_params,
                        event_callback=event_cb,
                        run_dir=run_dir,
                        stop_after=stop_after_val,
                        force_rerun_from=force_rerun_stage if force_rerun_stage != "None" else None,
                    )
                    final_snapshot = app_services.load_run_snapshot(run_dir) if run_dir else None
                    with dashboard_placeholder.container():
                        app_services._render_run_dashboard(final_snapshot)
                    status.update(
                        label=f"Processing finished at target: {stop_after_val}",
                        state="complete",
                    )

                store_processing_session_state(
                    st.session_state,
                    results=results,
                    validated_params=validated_params,
                    image_shape=image.shape,
                    dataset_name=uploaded_file.name,
                    run_dir=run_dir,
                    final_snapshot=final_snapshot,
                )
                app_services._render_run_dashboard(final_snapshot)
                if stop_after_val != "network":
                    st.warning(
                        f"⚠️ Pipeline halted early at '{stop_after_val}'. Downstream results (if any) are not available."
                    )
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success("🎉 Processing stage completed successfully!")
                st.markdown("</div>", unsafe_allow_html=True)
                processing_metrics = summarize_processing_metrics(results)
                col1, col2, col3, col4 = st.columns(4, gap="small", vertical_alignment="center")
                with col1:
                    st.metric(
                        "Vertices Found",
                        processing_metrics["vertices"] if "vertices" in results else "N/A",
                        help="Total vertices detected in the volume",
                    )
                with col2:
                    st.metric(
                        "Edges Extracted",
                        processing_metrics["edges"] if "edges" in results else "N/A",
                        help="Number of vessel segments traced",
                    )
                with col3:
                    st.metric(
                        "Network Strands",
                        processing_metrics["strands"] if "network" in results else "N/A",
                        help="Connected components in the network",
                    )
                with col4:
                    st.metric(
                        "Bifurcations",
                        processing_metrics["bifurcations"] if "network" in results else "N/A",
                        help="Detected branching points",
                    )
            except Exception as exc:
                st.error(f"❌ Processing failed: {exc!s}")
    else:
        st.info("👆 Please upload a TIFF file to begin processing")
