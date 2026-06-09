# SLAVV Python Tutorial

Quick walkthrough from setup to a processed vascular network.

## 1. Setup

Python 3.11+ required. Install the package:

```powershell
pip install -e ".[app]"
```

## 2. Generate Sample Data

No TIFF volume handy? Generate a synthetic vessel:

```python
import tifffile
from slavv_python.utils.synthetic import generate_synthetic_vessel_volume

# Create a 64x64x64 volume with a central vessel
volume = generate_synthetic_vessel_volume(shape=(64, 64, 64), vessel_radius=5.0)
tifffile.imwrite("sample_volume.tif", volume.astype('float32'))
```

## 3. Run the Pipeline

```powershell
slavv run -i sample_volume.tif -o output_folder --export csv json
```

Stages: Energy (Hessian enhancement) → Vertices (seed discovery) → Edges (watershed tracing) → Network (graph assembly).

## 4. Analyze Results

```powershell
slavv analyze -i output_folder/network.json
```

## 5. Visualize

```powershell
slavv plot -i output_folder/network.json -o my_plots.html
```

Open `my_plots.html` in a browser.

## 6. Interactive Curation (Optional)

```powershell
slavv-app
```

---

For more details, see the [Reference Documentation](README.md).
