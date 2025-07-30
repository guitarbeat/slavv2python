# SLAVV2Python - To-Do List

This document outlines planned improvements and added features for the SLAVV2Python project.

## 1. Core SLAVV Algorithm Implementation

- [x] Complete the implementation of `extract_vertices` in `src/vectorization_core.py`.
- [x] Complete the implementation of `_generate_edge_directions` in `src/vectorization_core.py`.
- [x] Complete the implementation of `_trace_edge` in `src/vectorization_core.py`.
- [x] Complete the implementation of `_find_terminal_vertex` in `src/vectorization_core.py`.
- [x] Complete the implementation of `_near_vertex` in `src/vectorization_core.py`.
- [x] Complete the implementation of `extract_edges` in `src/vectorization_core.py`.
- [x] Complete the implementation of `construct_network` in `src/vectorization_core.py`.
- [ ] Ensure all core functions in `src/vectorization_core.py` accurately reflect the MATLAB algorithm.

## 2. ML Curation

- [ ] Integrate actual machine learning models for vertex and edge curation in `src/ml_curator.py`.
- [ ] Provide options for users to upload their own pre-trained models or training data.
- [ ] Develop a clear workflow for training and deploying ML models within the Streamlit app.

## 3. Visualization Enhancements

- [ ] Refine 2D and 3D network visualizations for better clarity and interactivity.
- [ ] Add more visualization options (e.g., coloring by other metrics, interactive slicing for 3D data).
- [ ] Implement visualization for the energy field, allowing users to select slices.

## 4. Error Handling and Robustness

- [ ] Improve error handling for file uploads (e.g., non-TIFF files, corrupted TIFFs).
- [ ] Enhance parameter validation with more specific error messages and suggestions.
- [ ] Implement robust handling for edge cases in image processing and network construction.

## 5. Performance Optimization

- [ ] Profile the core SLAVV algorithm functions to identify performance bottlenecks.
- [ ] Explore using Numba or Cython for computationally intensive parts of the code.
- [ ] Optimize data structures and algorithms for better memory usage and speed.

## 6. Testing

- [ ] Develop comprehensive unit tests for all functions in `src/vectorization_core.py`, `src/ml_curator.py`, and `src/visualization.py`.
- [ ] Implement integration tests for the Streamlit application to ensure end-to-end functionality.
- [ ] Set up a continuous integration (CI) pipeline for automated testing.

## 7. Documentation

- [ ] Add detailed docstrings to all functions and classes in the codebase.
- [ ] Expand the `README.md` with more detailed setup instructions, usage examples, and troubleshooting tips.
- [ ] Create a dedicated `docs` directory for comprehensive user and developer documentation.

## 8. User Experience (UX) Enhancements

- [ ] Improve the responsiveness and layout of the Streamlit application for different screen sizes.
- [ ] Add tooltips and help texts for all input parameters and metrics.
- [ ] Implement a progress tracker for long-running operations.
- [ ] Provide clear feedback to the user at each step of the processing pipeline.

## 9. Export Formats

- [ ] Ensure full support for all export formats mentioned in the original MATLAB SLAVV documentation (VMV, CASX, MAT, CSV, JSON).
- [ ] Implement robust export functionalities for each format.



