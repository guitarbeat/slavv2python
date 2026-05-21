# SLAVV Python Tutorial

Welcome to SLAVV (Segmentation-Less, Automated, Vascular Vectorization). This tutorial will guide you through your first vascular extraction using the native Python pipeline.

## 1. Setup

Ensure you have Python 3.11+ and the package installed:

```powershell
pip install -e ".[app]"
```

## 2. Generate Sample Data

If you don't have a TIFF volume handy, you can generate a synthetic vessel for testing:

```python
import tifffile
from slavv_python.utils.synthetic import generate_synthetic_vessel_volume

# Create a 64x64x64 volume with a central vessel
volume = generate_synthetic_vessel_volume(shape=(64, 64, 64), vessel_radius=5.0)
tifffile.imwrite("sample_volume.tif", volume.astype('float32'))
```

## 3. Run the Pipeline

Execute the `run` command using the default `paper` profile:

```powershell
slavv run -i sample_volume.tif -o output_folder --export csv json
```

This will perform:
- **Energy**: Multiscale Hessian enhancement
- **Vertices**: Seed point discovery
- **Edges**: Watershed-guided vessel tracing
- **Network**: Graph assembly and strand smoothing

## 4. Analyze Results

Print a summary of the vascular network topology and statistics:

```powershell
slavv analyze -i output_folder/network.json
```

## 5. Visualize

Generate interactive 3D plots of the extracted network:

```powershell
slavv plot -i output_folder/network.json -o my_plots.html
```

Open `my_plots.html` in any web browser to explore the results.

## 6. Interactive Curation (Optional)

For more complex datasets, launch the Streamlit app to manually review and curate the network:

```powershell
slavv-app
```

---

For more details, see the [Reference Documentation](../README.md).
