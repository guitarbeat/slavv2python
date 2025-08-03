# Code Analysis Report

## Summary
This report contains the results of running copy-paste detection tools and dead code analysis on the codebase.

## Copy-Paste Detection Results

### PMD CPD (Copy/Paste Detector) - Python Files

The following copy-paste duplications were found in Python files:

#### 1. Neighborhood Calculation Logic (12 lines, 100 tokens)
- **File 1**: `slavv-streamlit/src/ml_curator.py` (lines 90-102)
- **File 2**: `slavv-streamlit/src/ml_curator.py` (lines 616-628)
- **Code**: Neighborhood size calculation and local energy extraction logic

#### 2. Success Message and Metrics Display (10 lines, 79 tokens)
- **File 1**: `slavv-streamlit/app.py` (lines 568-579)
- **File 2**: `slavv-streamlit/app.py` (lines 636-647)
- **Code**: Success message and metrics display with columns

#### 3. Strand Length Calculation (14 lines, 77 tokens)
- **File 1**: `slavv-streamlit/src/vectorization_core.py` (lines 598-611)
- **File 2**: `slavv-streamlit/src/visualization.py` (lines 356-369)
- **Code**: Strand length calculation logic

#### 4. 2D Network Plotting (8 lines, 58 tokens)
- **File 1**: `slavv-streamlit/app.py` (lines 733-740)
- **File 2**: `slavv-streamlit/app.py` (lines 741-748)
- **Code**: 2D network plotting with plotly

#### 5. Path Length Calculation (9 lines, 57 tokens)
- **File 1**: `slavv-streamlit/src/ml_curator.py` (lines 542-550)
- **File 2**: `slavv-streamlit/src/visualization.py` (lines 653-660)
- **Code**: Path length calculation method

#### 6. Path Length Calculation (7 lines, 56 tokens) - Multiple instances
- **Files**: `slavv-streamlit/src/ml_curator.py` (lines 542-548, 731-737), `slavv-streamlit/src/visualization.py` (lines 653-658)
- **Code**: Path length calculation method (appears in multiple locations)

#### 7. ML Results Dictionary (9 lines, 55 tokens)
- **File 1**: `slavv-streamlit/src/ml_curator.py` (lines 288-296)
- **File 2**: `slavv-streamlit/src/ml_curator.py` (lines 352-360)
- **Code**: Machine learning results dictionary structure

#### 8. Edge Plotting Logic (10 lines, 54 tokens)
- **File 1**: `slavv-streamlit/src/visualization.py` (lines 76-85)
- **File 2**: `slavv-streamlit/src/visualization.py` (lines 198-207)
- **Code**: Edge plotting with coordinate transformation

#### 9. Vertex Scatter Plot (9 lines, 53 tokens)
- **File 1**: `slavv-streamlit/src/visualization.py` (lines 131-139)
- **File 2**: `slavv-streamlit/src/visualization.py` (lines 254-262)
- **Code**: Vertex scatter plot configuration

#### 10. Vertex Marker Configuration (12 lines, 53 tokens)
- **File 1**: `slavv-streamlit/src/visualization.py` (lines 140-151)
- **File 2**: `slavv-streamlit/src/visualization.py` (lines 263-274)
- **Code**: Vertex marker configuration with hover templates

### jscpd Results

jscpd found similar duplications with additional details:

#### Clone 1 (8 lines, 78 tokens)
- `slavv-streamlit/src/visualization.py` [254:2 - 262:10]
- `slavv-streamlit/src/visualization.py` [131:2 - 139:8]

#### Clone 2 (11 lines, 87 tokens)
- `slavv-streamlit/src/visualization.py` [263:9 - 274:2]
- `slavv-streamlit/src/visualization.py` [140:9 - 151:7]

#### Clone 3 (10 lines, 124 tokens)
- `slavv-streamlit/src/vectorization_core.py` [601:5 - 611:15]
- `slavv-streamlit/src/visualization.py` [359:9 - 369:4]

#### Clone 4 (8 lines, 83 tokens)
- `slavv-streamlit/src/ml_curator.py` [352:9 - 360:29]
- `slavv-streamlit/src/ml_curator.py` [288:9 - 296:24]

#### Clone 5 (8 lines, 88 tokens)
- `slavv-streamlit/src/ml_curator.py` [542:5 - 550:11]
- `slavv-streamlit/src/visualization.py` [653:5 - 660:12]

#### Clone 6 (12 lines, 167 tokens)
- `slavv-streamlit/src/ml_curator.py` [616:2 - 628:9]
- `slavv-streamlit/src/ml_curator.py` [90:7 - 102:16]

#### Clone 7 (6 lines, 81 tokens)
- `slavv-streamlit/src/ml_curator.py` [731:5 - 737:2]
- `slavv-streamlit/src/visualization.py` [653:5 - 658:2]

#### Clone 8 (11 lines, 110 tokens)
- `slavv-streamlit/app.py` [636:26 - 647:19]
- `slavv-streamlit/app.py` [568:33 - 579:5]

#### Clone 9 (7 lines, 82 tokens)
- `slavv-streamlit/app.py` [741:16 - 748:19]
- `slavv-streamlit/app.py` [733:16 - 740:13]

## Dead Code Analysis (Vulture Results)

### High Confidence Issues (90% confidence)

#### Unused Imports
- `slavv-streamlit/app.py:7`: unused import 'io'
- `slavv-streamlit/src/visualization.py:13`: unused import 'plt'
- `slavv-streamlit/src/visualization.py:14`: unused import 'sns'

### Medium Confidence Issues (60% confidence)

#### Unused Functions and Methods
- `slavv-streamlit/app.py:77`: unused function 'main'
- `slavv-streamlit/src/ml_curator.py:231`: unused method 'train_vertex_classifier'
- `slavv-streamlit/src/ml_curator.py:304`: unused method 'train_edge_classifier'
- `slavv-streamlit/src/ml_curator.py:444`: unused method 'save_models'
- `slavv-streamlit/src/ml_curator.py:460`: unused method 'load_models'
- `slavv-streamlit/src/ml_curator.py:478`: unused method 'generate_training_data'
- `slavv-streamlit/src/visualization.py:544`: unused method 'create_summary_dashboard'
- `slavv-streamlit/src/visualization.py:622`: unused method 'export_network_data'

#### Unused Variables
- `slavv-streamlit/app.py:586`: unused variable 'vertex_curation_method'
- `slavv-streamlit/app.py:599`: unused variable 'edge_curation_method'
- `slavv-streamlit/app.py:717`: unused variable 'opacity'
- `slavv-streamlit/app.py:720`: unused variable 'camera_angle'
- `slavv-streamlit/src/ml_curator.py:286`: unused variable 'y_pred_proba'

#### Unused Attributes
- `slavv-streamlit/src/ml_curator.py:39`: unused attribute 'is_trained'
- `slavv-streamlit/src/visualization.py:29`: unused attribute 'color_schemes'

### Syntax Issues
- `slavv-streamlit/src/vectorization_core.py:443`: unexpected indent at "break"

## Recommendations

### 1. Refactor Duplicated Code
- Extract common neighborhood calculation logic into a utility function
- Create shared visualization utilities for plotting networks
- Implement a common path length calculation utility
- Consolidate ML results dictionary creation

### 2. Clean Up Dead Code
- Remove unused imports (io, plt, sns)
- Remove or implement unused functions and methods
- Clean up unused variables
- Fix the syntax error in vectorization_core.py

### 3. Code Organization
- Consider creating a shared utilities module for common operations
- Implement proper abstraction for visualization components
- Add proper error handling for the syntax issue

### 4. Testing
- Ensure that removing dead code doesn't break functionality
- Test refactored code thoroughly after consolidation
- Add unit tests for extracted utility functions

## Tools Used
- **PMD CPD**: Copy-paste detection for Java and Python
- **jscpd**: Multi-language copy-paste detection
- **Vulture**: Dead code detection for Python

## Detection Statistics
- **Total Copy-Paste Duplications**: 19 instances found
- **Dead Code Issues**: 15 instances found
- **Syntax Issues**: 1 instance found
- **Detection Time**: ~316ms (jscpd)