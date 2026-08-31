"""Image processing page for the SLAVV Streamlit app."""

from __future__ import annotations

from importlib import util

import numpy as np
import streamlit as st

from slavv_python.engine import SlavvPipeline
from slavv_python.interface.streamlit.navigation import switch_to
from slavv_python.interface.streamlit.services import app as app_services
from slavv_python.interface.streamlit.state.processing import (
    load_processing_snapshot,
    store_processing_session_state,
    summarize_processing_metrics,
)
from slavv_python.interface.streamlit.state.sample_data import (
    SAMPLE_TIFF_OPTIONS,
    build_sample_tiff,
)
from slavv_python.utils import validate_parameters
from slavv_python.workflows.profiles import PIPELINE_PROFILE_CHOICES, get_pipeline_profile_defaults


def available_public_energy_methods() -> list[str]:
    """Return Energy backends the current interpreter can import."""
    methods = ["hessian", "frangi", "sato"]
    if util.find_spec("SimpleITK") is not None:
        methods.append("simpleitk_objectness")
    if util.find_spec("cupy") is not None:
        methods.append("cupy_hessian")
    return methods


_PROFILE_WIDGET_DEFAULTS = {
    "processing_microns_per_voxel_y": lambda defaults: float(defaults["microns_per_voxel"][0]),
    "processing_microns_per_voxel_x": lambda defaults: float(defaults["microns_per_voxel"][1]),
    "processing_microns_per_voxel_z": lambda defaults: float(defaults["microns_per_voxel"][2]),
    "processing_approximating_psf": lambda defaults: bool(defaults["approximating_PSF"]),
    "processing_numerical_aperture": lambda defaults: float(defaults["numerical_aperture"]),
    "processing_excitation_wavelength": lambda defaults: float(
        defaults["excitation_wavelength_in_microns"]
    ),
    "processing_sample_index_of_refraction": lambda defaults: float(
        defaults["sample_index_of_refraction"]
    ),
    "processing_radius_smallest": lambda defaults: float(
        defaults["radius_of_smallest_vessel_in_microns"]
    ),
    "processing_radius_largest": lambda defaults: float(
        defaults["radius_of_largest_vessel_in_microns"]
    ),
    "processing_scales_per_octave": lambda defaults: float(defaults["scales_per_octave"]),
    "processing_energy_upper_bound": lambda defaults: float(defaults["energy_upper_bound"]),
    "processing_space_strel_apothem": lambda defaults: int(defaults["space_strel_apothem"]),
    "processing_space_strel_apothem_edges": lambda defaults: int(
        defaults["space_strel_apothem_edges"]
    ),
    "processing_length_dilation_ratio": lambda defaults: float(defaults["length_dilation_ratio"]),
    "processing_number_of_edges_per_vertex": lambda defaults: int(
        defaults["number_of_edges_per_vertex"]
    ),
    "processing_max_voxels_per_node": lambda defaults: int(defaults["max_voxels_per_node_energy"]),
    "processing_gaussian_to_ideal_ratio": lambda defaults: float(
        defaults["gaussian_to_ideal_ratio"]
    ),
    "processing_spherical_to_annular_ratio": lambda defaults: float(
        defaults["spherical_to_annular_ratio"]
    ),
    "processing_energy_projection_mode": lambda defaults: str(defaults["energy_projection_mode"]),
    "processing_energy_method": lambda defaults: str(defaults["energy_method"]),
    "processing_edge_method": lambda defaults: str(defaults["edge_method"]),
    "processing_step_size_per_origin_radius": lambda defaults: float(
        defaults["step_size_per_origin_radius"]
    ),
    "processing_max_edge_length_per_origin_radius": lambda defaults: float(
        defaults["max_edge_length_per_origin_radius"]
    ),
    "processing_max_edge_energy": lambda defaults: float(defaults["max_edge_energy"]),
    "processing_min_hair_length": lambda defaults: float(defaults["min_hair_length_in_microns"]),
}


def _sync_processing_profile_defaults(profile: str) -> dict[str, object]:
    """Reset paper-profile-backed widget defaults when the selected profile changes."""
    defaults = get_pipeline_profile_defaults(profile)
    previous_profile = st.session_state.get("processing_profile_applied")
    if previous_profile != profile:
        for state_key, builder in _PROFILE_WIDGET_DEFAULTS.items():
            st.session_state[state_key] = builder(defaults)
        st.session_state["processing_profile_applied"] = profile
    else:
        for state_key, builder in _PROFILE_WIDGET_DEFAULTS.items():
            st.session_state.setdefault(state_key, builder(defaults))
    return defaults


def show_processing_page() -> None:
    """Display the image processing page."""
    st.markdown('<h2 class="section-header">Process a TIFF volume</h2>', unsafe_allow_html=True)

    with st.sidebar:
        st.divider()
        st.subheader("Processing input")
        input_source = st.radio(
            "Input source",
            ("Upload your TIFF", "Use a built-in sample"),
            key="processing_input_source",
        )

    input_bytes: bytes | None = None
    dataset_name: str | None = None
    sample_id = ""
    if input_source == "Upload your TIFF":
        st.subheader("Input TIFF")
        uploaded_file = st.file_uploader(
            "Choose a TIFF file",
            type=["tif", "tiff"],
            help="Upload a 3D grayscale TIFF image of vascular structures",
        )
        if uploaded_file is not None:
            input_bytes = uploaded_file.getvalue()
            dataset_name = uploaded_file.name
            st.success(f"Loaded input: {dataset_name}")
            st.caption(f"{len(input_bytes) / 1024 / 1024:.2f} MB · {uploaded_file.type}")
    else:
        with st.sidebar:
            sample_id = st.selectbox(
                "Sample geometry",
                options=list(SAMPLE_TIFF_OPTIONS),
                format_func=lambda value: SAMPLE_TIFF_OPTIONS[value],
                key="processing_sample_id",
            )
        sample = build_sample_tiff(sample_id)
        input_bytes = sample.tiff_bytes
        dataset_name = sample.name
        preview = app_services.cached_load_tiff_bytes(input_bytes)
        preview_col, detail_col = st.columns([2, 1], gap="large", vertical_alignment="center")
        with preview_col:
            st.image(
                np.max(preview, axis=2),
                caption="Maximum-intensity projection of the sample volume",
                width="stretch",
                clamp=True,
            )
        with detail_col:
            st.subheader("Built-in sample TIFF")
            st.write(sample.description)
            st.caption(
                f"TIFF shape (Z, Y, X): {sample.shape_zyx[0]} x "
                f"{sample.shape_zyx[1]} x {sample.shape_zyx[2]}"
            )
            st.download_button(
                "Download sample TIFF",
                data=input_bytes,
                file_name=dataset_name,
                mime="image/tiff",
                icon=":material/download:",
            )

    st.markdown('<h3 class="section-header">Processing settings</h3>', unsafe_allow_html=True)
    with st.popover("Parameter tips", width=300):
        st.write(
            "Use the tabs below to adjust microscopy, vessel size, processing, "
            "and advanced options. Defaults are provided for typical datasets."
        )

    pipeline_profile = st.selectbox(
        "Pipeline profile",
        options=list(PIPELINE_PROFILE_CHOICES),
        format_func=lambda profile: {
            "paper": "Paper Path — tracing discovery (recommended)",
            "matlab_compat": "MATLAB-compat defaults — watershed in Advanced",
        }[profile],
        index=0,
        help=(
            "Paper Path uses tracing-based edge discovery for routine processing. "
            "MATLAB-compat keeps legacy defaults; choose Watershed Discovery under "
            "Advanced for Exact Route / certification-style runs."
        ),
        key="processing_pipeline_profile",
    )
    _sync_processing_profile_defaults(pipeline_profile)
    if input_source == "Use a built-in sample":
        sample_defaults_key = f"{pipeline_profile}:{sample_id}"
        if st.session_state.get("processing_sample_defaults_applied") != sample_defaults_key:
            st.session_state["processing_radius_smallest"] = 1.0
            st.session_state["processing_radius_largest"] = 5.0
            st.session_state["processing_scales_per_octave"] = 1.0
            st.session_state["processing_sample_defaults_applied"] = sample_defaults_key
    if pipeline_profile == "paper":
        st.caption(
            "Standard Python workflow (Paper Path): uses tracing-based edge discovery "
            "and the published Hessian projection. Recommended for routine processing."
        )
    else:
        st.caption(
            "MATLAB-compatible parameter defaults on the same Python pipeline. "
            "The MATLAB-faithful watershed method is available under Advanced → Edge method."
        )

    tab1, tab2, tab3, tab4 = st.tabs(["Microscopy", "Vessel sizes", "Processing", "Advanced"])

    with tab1:
        st.markdown("#### Microscopy settings")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            microns_per_voxel_y = st.number_input(
                "Y voxel size (µm)",
                min_value=0.01,
                max_value=10.0,
                value=float(st.session_state["processing_microns_per_voxel_y"]),
                step=0.01,
                key="processing_microns_per_voxel_y",
                help="Physical size of one voxel in Y dimension. (MATLAB: microns_per_voxel(1))",
            )
            microns_per_voxel_x = st.number_input(
                "X voxel size (µm)",
                min_value=0.01,
                max_value=10.0,
                value=float(st.session_state["processing_microns_per_voxel_x"]),
                step=0.01,
                key="processing_microns_per_voxel_x",
                help="Physical size of one voxel in X dimension. (MATLAB: microns_per_voxel(2))",
            )
            microns_per_voxel_z = st.number_input(
                "Z voxel size (µm)",
                min_value=0.01,
                max_value=10.0,
                value=float(st.session_state["processing_microns_per_voxel_z"]),
                step=0.01,
                key="processing_microns_per_voxel_z",
                help="Physical size of one voxel in Z dimension. (MATLAB: microns_per_voxel(3))",
            )
        with col2:
            approximating_PSF = st.checkbox(
                "Correct for microscope blur (PSF)",
                value=bool(st.session_state["processing_approximating_psf"]),
                key="processing_approximating_psf",
                help="Account for the microscope point-spread function (PSF) using the Zipfel et al. model.",
            )
            if approximating_PSF:
                numerical_aperture = st.number_input(
                    "Numerical aperture",
                    min_value=0.1,
                    max_value=2.0,
                    value=float(st.session_state["processing_numerical_aperture"]),
                    step=0.01,
                    key="processing_numerical_aperture",
                    help="Numerical aperture of the microscope objective. (MATLAB: numerical_aperture)",
                )
                excitation_wavelength = st.number_input(
                    "Excitation wavelength (µm)",
                    min_value=0.4,
                    max_value=3.0,
                    value=float(st.session_state["processing_excitation_wavelength"]),
                    step=0.1,
                    key="processing_excitation_wavelength",
                    help="Laser excitation wavelength. A typical two-photon range is 0.7-3.0 µm.",
                )
                if not (0.7 <= excitation_wavelength <= 3.0):
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.warning(
                        "Excitation wavelength is outside the typical two-photon range "
                        "(0.7-3.0 µm). Please verify this value."
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                sample_index_of_refraction = st.number_input(
                    "Sample refractive index",
                    min_value=1.0,
                    max_value=2.0,
                    value=float(st.session_state["processing_sample_index_of_refraction"]),
                    step=0.01,
                    key="processing_sample_index_of_refraction",
                    help="Refractive index of the sample medium (e.g., 1.33 for water). (MATLAB: sample_index_of_refraction)",
                )

    with tab2:
        st.markdown("#### Vessel size settings")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            radius_smallest = st.number_input(
                "Smallest vessel radius (µm)",
                min_value=0.1,
                max_value=100.0,
                value=float(st.session_state["processing_radius_smallest"]),
                step=0.1,
                key="processing_radius_smallest",
                help="Radius of the smallest vessel to be detected in microns. (MATLAB: radius_of_smallest_vessel_in_microns)",
            )
            radius_largest = st.number_input(
                "Largest vessel radius (µm)",
                min_value=1.0,
                max_value=500.0,
                value=float(st.session_state["processing_radius_largest"]),
                step=1.0,
                key="processing_radius_largest",
                help="Radius of the largest vessel to be detected in microns. (MATLAB: radius_of_largest_vessel_in_microns)",
            )
            if radius_largest <= radius_smallest:
                st.error("Largest radius must be greater than smallest radius")
        with col2:
            scales_per_octave = st.number_input(
                "Scales per octave",
                min_value=0.5,
                max_value=5.0,
                value=float(st.session_state["processing_scales_per_octave"]),
                step=0.1,
                key="processing_scales_per_octave",
                help="Number of vessel sizes to detect per doubling of the radius cubed. (MATLAB: scales_per_octave)",
            )
            if radius_largest > radius_smallest:
                volume_ratio = (radius_largest / radius_smallest) ** 3
                n_scales = int(np.log(volume_ratio) / np.log(2) * scales_per_octave) + 3
                st.info(f"This will generate approximately {n_scales} scales")

    with tab3:
        st.markdown("#### Processing options")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            energy_upper_bound = st.number_input(
                "Energy upper bound",
                min_value=-10.0,
                max_value=0.0,
                value=float(st.session_state["processing_energy_upper_bound"]),
                step=0.1,
                key="processing_energy_upper_bound",
                help="Maximum energy value for vertex detection (negative values). (MATLAB: energy_upper_bound)",
            )
            space_strel_apothem = st.number_input(
                "Spatial structuring element",
                min_value=1,
                max_value=10,
                value=int(st.session_state["processing_space_strel_apothem"]),
                step=1,
                key="processing_space_strel_apothem",
                help="Minimum spacing between detected vertices (in voxels). (MATLAB: space_strel_apothem)",
            )
            length_dilation_ratio = st.number_input(
                "Length dilation ratio",
                min_value=0.1,
                max_value=5.0,
                value=float(st.session_state["processing_length_dilation_ratio"]),
                step=0.1,
                key="processing_length_dilation_ratio",
                help="Ratio of rendering length to detection length for volume exclusion. (MATLAB: length_dilation_ratio)",
            )
        with col2:
            number_of_edges_per_vertex = st.number_input(
                "Edges per vertex",
                min_value=1,
                max_value=10,
                value=int(st.session_state["processing_number_of_edges_per_vertex"]),
                step=1,
                key="processing_number_of_edges_per_vertex",
                help="Maximum number of edge traces per seed vertex. (MATLAB: number_of_edges_per_vertex)",
            )
            space_strel_apothem_edges = st.number_input(
                "Edge exclusion spacing",
                min_value=1,
                max_value=10,
                value=int(st.session_state["processing_space_strel_apothem_edges"]),
                step=1,
                key="processing_space_strel_apothem_edges",
                help="Minimum spacing used by edge exclusion logic. (MATLAB: space_strel_apothem_edges)",
            )
            max_voxels_per_node = st.number_input(
                "Max voxels per node",
                min_value=1000,
                max_value=1000000,
                value=int(st.session_state["processing_max_voxels_per_node"]),
                step=1000,
                key="processing_max_voxels_per_node",
                help="Maximum voxels per computational node for parallel processing. (MATLAB: max_voxels_per_node_energy)",
            )

    with tab4:
        st.markdown("#### Advanced settings")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            gaussian_to_ideal_ratio = st.slider(
                "Gaussian to ideal ratio",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state["processing_gaussian_to_ideal_ratio"]),
                step=0.1,
                key="processing_gaussian_to_ideal_ratio",
                help="Standard deviation of the Gaussian kernel per the total object length for objects that are much larger than the PSF. (MATLAB: gaussian_to_ideal_ratio)",
            )
            spherical_to_annular_ratio = st.slider(
                "Spherical to annular ratio",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state["processing_spherical_to_annular_ratio"]),
                step=0.1,
                key="processing_spherical_to_annular_ratio",
                help="Weighting factor of the spherical pulse over the combined weights of spherical and annular pulses. (MATLAB: spherical_to_annular_ratio)",
            )
        with col2:
            energy_method_options = available_public_energy_methods()
            current_energy = str(st.session_state.get("processing_energy_method", "hessian"))
            if current_energy not in energy_method_options:
                st.session_state["processing_energy_method"] = energy_method_options[0]
            energy_method = st.selectbox(
                "Energy method",
                options=energy_method_options,
                key="processing_energy_method",
                help=(
                    "Energy backend. Optional backends (SimpleITK, CuPy) appear only "
                    "when those packages import in this environment."
                ),
            )
            edge_method_labels = {
                "tracing": "Tracing Discovery (Paper Path)",
                "watershed": "Watershed Discovery (Exact Route)",
            }
            current_edge = str(st.session_state.get("processing_edge_method", "tracing"))
            if current_edge not in edge_method_labels:
                st.session_state["processing_edge_method"] = "tracing"
            edge_method = st.selectbox(
                "Edge method",
                options=list(edge_method_labels),
                format_func=lambda key: edge_method_labels[key],
                key="processing_edge_method",
                help=(
                    "Tracing Discovery is the public Paper Path. "
                    "Watershed Discovery is the Exact Route / certification method."
                ),
            )
            energy_projection_mode = st.selectbox(
                "Energy projection mode",
                options=["matlab", "paper"],
                index=["matlab", "paper"].index(
                    st.session_state["processing_energy_projection_mode"]
                ),
                key="processing_energy_projection_mode",
                help=(
                    "Projection rule for the default Hessian energy stack. "
                    "'matlab' follows the released MATLAB minimum projection, "
                    "while 'paper' uses the published blended scale estimate."
                ),
            )
            step_size_per_origin_radius = st.number_input(
                "Step size ratio",
                min_value=0.1,
                max_value=5.0,
                value=float(st.session_state["processing_step_size_per_origin_radius"]),
                step=0.1,
                key="processing_step_size_per_origin_radius",
                help="Edge tracing step size relative to origin vertex radius. (MATLAB: step_size_per_origin_radius)",
            )
            max_edge_length_per_origin_radius = st.number_input(
                "Max edge length ratio",
                min_value=1.0,
                max_value=200.0,
                value=float(st.session_state["processing_max_edge_length_per_origin_radius"]),
                step=1.0,
                key="processing_max_edge_length_per_origin_radius",
                help="Maximum trace length relative to origin vertex radius. (MATLAB: max_edge_length_per_origin_radius)",
            )
            max_edge_energy = st.number_input(
                "Max edge energy",
                min_value=-10.0,
                max_value=0.0,
                value=float(st.session_state["processing_max_edge_energy"]),
                step=0.1,
                key="processing_max_edge_energy",
                help="Maximum energy threshold for edge tracing. (MATLAB: max_edge_energy)",
            )
            min_hair_length_in_microns = st.number_input(
                "Minimum terminal-branch length (µm)",
                min_value=0.0,
                max_value=1000.0,
                value=float(st.session_state["processing_min_hair_length"]),
                step=0.5,
                key="processing_min_hair_length",
                help="Minimum terminal hair length preserved during cleanup. (MATLAB: min_hair_length_in_microns)",
            )

    st.markdown('<h3 class="section-header">Processing</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        stop_after_options = {
            "Energy only": "energy",
            "Energy + Vertices": "vertices",
            "Energy + Vertices + Edges": "edges",
            "Full pipeline (Network)": "network",
        }
        stop_after_selection = st.selectbox(
            "Run through",
            options=list(stop_after_options.keys()),
            index=3,
            help="Stop the pipeline early after completing this stage. Useful for tweaking parameters.",
        )
        stop_after_val = stop_after_options[stop_after_selection]
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        force_rerun_options = {
            "Use available stage results": None,
            "Energy": "energy",
            "Vertices": "vertices",
            "Edges": "edges",
            "Network": "network",
        }
        force_rerun_selection = st.selectbox(
            "Recalculate from",
            options=list(force_rerun_options.keys()),
            index=0,
            help="Ignore cached results and recalculate from this stage onward.",
        )
        force_rerun_stage = force_rerun_options[force_rerun_selection]

    current_snapshot = load_processing_snapshot(
        st.session_state,
        snapshot_loader=app_services.load_run_snapshot,
    )
    if current_snapshot is not None:
        app_services._render_run_dashboard(
            current_snapshot,
            run_dir=st.session_state.get("current_run_dir"),
        )

    if input_bytes is not None and dataset_name is not None:
        if st.button("Run processing", type="primary", width=250):
            parameters = {
                "pipeline_profile": pipeline_profile,
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
                "space_strel_apothem_edges": space_strel_apothem_edges,
                "length_dilation_ratio": length_dilation_ratio,
                "number_of_edges_per_vertex": number_of_edges_per_vertex,
                "max_voxels_per_node_energy": max_voxels_per_node,
                "gaussian_to_ideal_ratio": gaussian_to_ideal_ratio,
                "spherical_to_annular_ratio": spherical_to_annular_ratio,
                "energy_method": energy_method,
                "energy_projection_mode": energy_projection_mode,
                "edge_method": edge_method,
                "step_size_per_origin_radius": step_size_per_origin_radius,
                "max_edge_length_per_origin_radius": max_edge_length_per_origin_radius,
                "max_edge_energy": max_edge_energy,
                "min_hair_length_in_microns": min_hair_length_in_microns,
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
                st.success("Settings validated")

                with st.status("Processing image...", expanded=True) as status:
                    status.update(label="Loading image...", state="running")
                    try:
                        image = app_services.cached_load_tiff_bytes(input_bytes)
                        st.success(f"Image loaded · shape {image.shape}")
                    except ValueError as exc:
                        st.error(f"Error loading TIFF file: {exc}")
                        st.stop()

                    processor = SlavvPipeline()
                    dashboard_placeholder = st.empty()
                    run_dir = app_services._build_processing_run_dir(
                        input_bytes,
                        validated_params,
                    )

                    def event_cb(event) -> None:
                        state = "complete" if event.status.startswith("completed") else "running"
                        label = event.detail or f"{event.stage} {int(event.stage_progress * 100)}%"
                        status.update(label=label, state=state)
                        with dashboard_placeholder.container():
                            app_services._render_run_dashboard(event.snapshot, run_dir=run_dir)

                    results = processor.run(
                        image,
                        validated_params,
                        event_callback=event_cb,
                        run_dir=run_dir,
                        stop_after=stop_after_val,
                        force_rerun_from=force_rerun_stage,
                    )
                    final_snapshot = app_services.load_run_snapshot(run_dir) if run_dir else None
                    with dashboard_placeholder.container():
                        app_services._render_run_dashboard(final_snapshot, run_dir=run_dir)
                    status.update(
                        label=f"Processing complete through {stop_after_val.title()}",
                        state="complete",
                    )

                store_processing_session_state(
                    st.session_state,
                    results=results,
                    validated_params=validated_params,
                    image_shape=image.shape,
                    dataset_name=dataset_name,
                    run_dir=run_dir,
                    final_snapshot=final_snapshot,
                    original_volume=image,
                )
                app_services._render_run_dashboard(final_snapshot, run_dir=run_dir)
                if stop_after_val != "network":
                    st.warning(
                        f"Processing stopped after {stop_after_val.title()}, as requested. "
                        "Later workflow steps remain unavailable."
                    )
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success("Processing complete")
                st.markdown("</div>", unsafe_allow_html=True)
                processing_metrics = summarize_processing_metrics(results)
                col1, col2, col3, col4 = st.columns(4, gap="small", vertical_alignment="center")
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Vertices",
                        processing_metrics["vertices"] if "vertices" in results else "N/A",
                        help="Total vertices detected in the volume",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Edges",
                        processing_metrics["edges"] if "edges" in results else "N/A",
                        help="Number of vessel segments traced",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Strands",
                        processing_metrics["strands"] if "network" in results else "N/A",
                        help="Connected components in the network",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Bifurcations",
                        processing_metrics["bifurcations"] if "network" in results else "N/A",
                        help="Detected branching points",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                next_columns = st.columns(2, gap="small")
                if "edges" in results and next_columns[0].button(
                    "Review in Curation",
                    type="primary",
                    icon=":material/edit_note:",
                    width="stretch",
                ):
                    switch_to("curation")
                if "network" in results and next_columns[1].button(
                    "Open Visualization",
                    icon=":material/view_in_ar:",
                    width="stretch",
                ):
                    switch_to("visualization")
            except Exception as exc:
                st.error(f"Processing failed: {exc!s}")
    else:
        st.info("Upload a TIFF or choose a built-in sample to begin processing.")
