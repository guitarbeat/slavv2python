# System Architecture

## 1. High-Level Context
The SLAVV2Python system is designed as a modular library (`slavv`) consumed by various interfaces (CLI, Web App, Scripts).

```mermaid
graph TD
    User([User / Researcher])
    
    subgraph "Interfaces"
        CLI[Command Line Interface]
        WebApp[Streamlit App]
        Jupyter[Jupyter Notebooks]
    end
    
    subgraph "Core Library (src/slavv)"
        API[SLAVVProcessor]
        IO[IO Utils]
        Algo[Vectorization Logic]
    end
    
    subgraph "Data Storage"
        FS[File System (.tif, .h5)]
    end

    User --> CLI
    User --> WebApp
    User --> Jupyter
    
    CLI --> API
    WebApp --> API
    Jupyter --> API
    
    API --> Algo
    API --> IO
    
    IO --> FS
```

## 2. Vectorization Pipeline Data Flow
The core transformation turns a raster volume (voxels) into a topological graph (nodes/edges).

```mermaid
flowchart LR
    A[Input Volume<br/>(TIFF/Numpy)] --> B{Energy Field<br/>Calculation}
    B -->|Potential Map| C[Vertex Extraction<br/>(Local Maxima)]
    C -->|Seed Points| D[Edge Tracing<br/>(Gradient Descent)]
    D -->|Raw Segments| E[Network Construction<br/>(Graph Logic)]
    E --> F[Vascular Graph<br/>(Nodes + Edges)]
    
    subgraph "Energy Module"
    B 
    end
    
    subgraph "Tracing Module"
    C
    D
    end
    
    subgraph "Graph Module"
    E
    end
```

## 3. Data Structure Evolution
As data moves through the pipeline, its representation changes:

1.  **Raster Phase (`image`, `energy`)**: 
    -   Type: `numpy.ndarray` (float32)
    -   Shape: `(Y, X, Z)`
    -   Memory: High (GBs)

2.  **Point Phase (`vertices`)**:
    -   Type: `dict` of Arrays
    -   `positions`: `(N, 3)` coordinates
    -   `scales`: `(N,)` radii indices

3.  **Trace Phase (`edges`)**:
    -   Type: `list[numpy.ndarray]`
    -   Each element is a `(M, 3)` polyline

4.  **Graph Phase (`network`)**:
    -   Type: Abstract Graph (Adjacency List + Node Attributes)
    -   Serialized as: `.h5` (MorphIO), `.vmv` (VessMorphoVis), or `.mat` (Legacy)

## 4. Design Principles

### A. Separation of Concerns
-   **Core Logic (`src/slavv`)**: Pure Python, few dependencies, no UI code.
-   **Visualization (`src/slavv/apps/web_app.py`, `external/blender_resources`)**: Handles rendering and user interaction.
-   This allows the core to run on headless HPC nodes without X11 forwarding.

### B. Stateless Processing
-   The `SLAVVProcessor` is designed to be largely stateless.
-   Pipeline stages (`process_image`) pass data dictionaries explicitly.
-   Checkpointing (optional) is the only state persistence mechanism.

### C. Explicit Configuration
-   No hidden globals. All physical parameters (`microns_per_voxel`, `scales`) must be passed in the `params` dictionary.
-   Defaults are handled in `utils.validate_parameters` but can always be overridden.
