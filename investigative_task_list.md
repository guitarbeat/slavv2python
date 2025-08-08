# Investigative Task List

## Priority 1: Critical Issues

### 1.1 Fix Syntax Error
- **Task**: Investigate and fix unexpected indent at line 443 in `slavv-streamlit/src/vectorization_core.py`
- **Impact**: High - Syntax errors prevent code execution
- **Effort**: Low (quick fix)
- **Status**: Completed — removed stray duplicated block, deduplicated helpers, recompiled successfully

### 1.2 Remove Unused Imports (High Confidence)
- **Task**: Remove unused imports identified by vulture (90% confidence)
  - `slavv-streamlit/app.py:7`: unused import 'io'
  - `slavv-streamlit/src/visualization.py:13`: unused import 'plt'
  - `slavv-streamlit/src/visualization.py:14`: unused import 'sns'
- **Impact**: Medium - Reduces code bloat and import overhead
- **Effort**: Low (simple deletions)
- **Action**: Verify imports are truly unused, then remove

## Priority 2: Code Duplication Investigation

### 2.1 Path Length Calculation Duplication
- **Task**: Investigate duplicated path length calculation methods
- **Files**: 
  - `slavv-streamlit/src/ml_curator.py` (lines 542-550, 731-737)
  - `slavv-streamlit/src/visualization.py` (lines 653-660)
- **Investigation Points**:
  - Are these methods functionally identical?
  - Can they be consolidated into a shared utility?
  - Are there any class-specific dependencies preventing consolidation?
- **Action**: Create shared utility function if possible

### 2.2 Neighborhood Calculation Logic
- **Task**: Investigate duplicated neighborhood calculation code
- **Files**: `slavv-streamlit/src/ml_curator.py` (lines 90-102, 616-628)
- **Investigation Points**:
  - Are the neighborhood calculations identical?
  - What are the input/output differences?
  - Can this be extracted to a common method?
- **Action**: Extract to utility function if logic is identical

### 2.3 Visualization Plotting Duplications
- **Task**: Investigate duplicated plotting code in visualization module
- **Files**: `slavv-streamlit/src/visualization.py` (multiple instances)
- **Investigation Points**:
  - Are the plotting configurations identical?
  - Can common plotting utilities be created?
  - Are there performance implications of consolidation?
- **Action**: Create shared plotting utilities

### 2.4 ML Results Dictionary Structure
- **Task**: Investigate duplicated ML results dictionary creation
- **Files**: `slavv-streamlit/src/ml_curator.py` (lines 288-296, 352-360)
- **Investigation Points**:
  - Are the dictionary structures identical?
  - Can a factory method be created?
  - Are there validation requirements?
- **Action**: Create factory method for ML results

## Priority 3: Dead Code Investigation

### 3.1 Unused Functions Investigation
- **Task**: Investigate unused functions and methods (60% confidence)
- **Files**: 
  - `slavv-streamlit/app.py:77`: unused function 'main'
  - `slavv-streamlit/src/ml_curator.py`: multiple unused methods
  - `slavv-streamlit/src/visualization.py`: unused methods
- **Investigation Points**:
  - Are these functions actually unused or false positives?
  - Are they intended for future use?
  - Should they be removed or implemented?
- **Action**: Verify usage patterns and decide on removal/implementation

### 3.2 Unused Variables Investigation
- **Task**: Investigate unused variables
- **Files**: `slavv-streamlit/app.py` (lines 586, 599, 717, 720)
- **Investigation Points**:
  - Are these variables truly unused?
  - Were they intended for debugging or future features?
  - Can they be safely removed?
- **Action**: Verify and clean up unused variables

### 3.3 Unused Attributes Investigation
- **Task**: Investigate unused class attributes
- **Files**: 
  - `slavv-streamlit/src/ml_curator.py:39`: unused attribute 'is_trained'
  - `slavv-streamlit/src/visualization.py:29`: unused attribute 'color_schemes'
- **Investigation Points**:
  - Are these attributes used in inheritance or reflection?
  - Were they intended for future features?
  - Can they be safely removed?
- **Action**: Verify usage and clean up

## Priority 4: Code Organization Investigation

### 4.1 Shared Utilities Assessment
- **Task**: Investigate potential for creating shared utilities
- **Areas**:
  - Mathematical calculations (path length, neighborhood)
  - Visualization components
  - Data processing utilities
  - ML result formatting
- **Investigation Points**:
  - What common patterns exist across modules?
  - Can utilities be shared without breaking encapsulation?
  - What are the performance implications?
- **Action**: Design and implement shared utility modules

### 4.2 Module Dependencies Analysis
- **Task**: Investigate module dependencies and coupling
- **Investigation Points**:
  - Are there circular dependencies?
  - Can modules be better organized?
  - Are there opportunities for better abstraction?
- **Action**: Refactor module structure if needed

## Priority 5: Performance and Quality Investigation

### 5.1 Code Quality Metrics
- **Task**: Investigate overall code quality
- **Areas**:
  - Cyclomatic complexity
  - Function length
  - Class cohesion
  - Test coverage
- **Investigation Points**:
  - Are there overly complex functions?
  - Is test coverage adequate?
  - Are there code smells beyond duplication?
- **Action**: Implement quality improvements

### 5.2 Performance Impact Assessment
- **Task**: Investigate performance implications of refactoring
- **Investigation Points**:
  - Will consolidation improve performance?
  - Are there memory usage implications?
  - Will shared utilities impact startup time?
- **Action**: Measure and optimize performance

## Priority 6: MATLAB Parity Follow-ups

- **Energy Field Parity**: Align kernel construction and PSF weighting with `get_energy_V202.m`
- **Vertex Extraction Parity**: Optimize volume exclusion and geometry to match `get_vertices_V200.m`
- **Edge Tracing Parity**: Implement gradient-descent ridge following and termination heuristics akin to `get_edges_V300.m`
- **Network Cleaning**: Add steps for hairs/orphans/cycles consistent with MATLAB scripts

Note: These items come from recent source checks against the original MATLAB repository.

## Investigation Guidelines

### For Each Task:
1. **Examine the specific code** in context
2. **Check for false positives** in static analysis
3. **Verify dependencies** and usage patterns
4. **Test thoroughly** before making changes
5. **Document decisions** and rationale
6. **Measure impact** of changes

### Risk Assessment:
- **Low Risk**: Removing unused imports, fixing syntax errors
- **Medium Risk**: Consolidating utility functions, removing unused variables
- **High Risk**: Refactoring core functionality, changing module structure

### Success Criteria:
- No functionality is broken
- Code is more maintainable
- Performance is maintained or improved
- Test coverage is adequate
- Documentation is updated

## Next Steps:
1. Start with Priority 1 tasks (critical issues)
2. Move to Priority 2 (code duplication) for high-impact improvements
3. Address Priority 3 (dead code) systematically
4. Plan Priority 4 (organization) for long-term improvements
5. Consider Priority 5 (quality) for ongoing maintenance