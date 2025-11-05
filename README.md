# GLCM Analyzer

![Python 3.7-3.13](https://img.shields.io/badge/python-3.7%20|%203.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12%20|%203.13-blue)

A high-performance Python package for Gray-Level Co-occurrence Matrix (GLCM) texture analysis of satellite imagery, optimized for large-scale urban and arid environments. Perfect for land cover classification, vegetation analysis, and urban mapping.

# Features

- 🚀 Parallel Processing: Handle massive satellite imagery (3500+ km²) efficiently

- 🎯 Smart Window Sizing: Automatically optimize GLCM window size for your specific data

- 📐 Vector Data Support: Work directly with GeoJSON, Shapefiles, and other vector formats

- 🌳 Vegetation Analysis: Specialized for tree vs. grass classification in arid urban areas

- 💫 Multiple Textures: Compute contrast, entropy, correlation, and other GLCM features

- 🐍 Python 3.7-3.13: Full compatibility across Python versions

# Installation

```bash
# Install from source
git clone https://github.com/DijoG/GLCM.git
cd glcm_analyzer
pip install -e .

# Or install dependencies directly
pip install rasterio scikit-image geopandas matplotlib tqdm pandas
```
## Package Structure

```text
GLCM/
├── glcm_analyzer/            # Python package
│ ├── init.py                 # Package initialization
│ ├── core.py                 # Core GLCM processing functions
│ ├── optimization.py         # Window size optimization algorithms
│ ├── vector_utils.py         # Vector data conversion utilities
│ └── utils.py                # General helper functions
├── examples/                 # Example scripts
│ ├── optimize_window_size.py # Window optimization example
│ └── run_parallel_glcm.py    # Full processing example
├── setup.py                  # Package configuration
├── README.md                 # This documentation
├── test_python313.py         # Python compatibility testing
└── diagnose_skimage.py       # scikit-image diagnostics
```

# Quick Start

## 1. Optimize Window Size
```python
from glcm_analyzer.optimization import optimize_window_size_from_vectors
from glcm_analyzer.vector_utils import get_vector_statistics

# Find the optimal GLCM window size for your data
results_df, optimal_ws = optimize_window_size_from_vectors(
    panchromatic_path="data/riyadh_pan_0.35m.tif",
    tree_vector_path="data/training/tree_polygons.geojson",
    grass_vector_path="data/training/grass_polygons.geojson", 
    output_dir="results/optimization",
    sample_km_side=10,  # 10km × 10km sample area
    max_workers=8
)

print(f"Optimal window size: {optimal_ws}x{optimal_ws} pixels")
```

## 2. Run GLCM Analysis
```python
from glcm_analyzer.core import hybrid_parallel_glcm

# Process entire dataset with optimal settings
hybrid_parallel_glcm(
    input_path="data/riyadh_pan_0.35m.tif",
    output_dir="results/glcm_textures",
    window_size=11,  # From optimization step
    chunk_size=2048,
    max_workers=12,
    metrics=['contrast', 'entropy', 'correlation']
)
```
# Complete Workflow
```python
from glcm_analyzer.optimization import optimize_window_size_from_vectors
from glcm_analyzer.core import hybrid_parallel_glcm
from glcm_analyzer.vector_utils import get_vector_statistics

# 1. Analyze your training data
print("Analyzing training data...")
tree_stats = get_vector_statistics("data/trees.geojson")
grass_stats = get_vector_statistics("data/grass.geojson")

# 2. Find optimal parameters
# OPTION 1:
print("Optimizing GLCM window size...")
results_df, optimal_ws = optimize_window_size_from_vectors(
    panchromatic_path="data/riyadh_pan_0.35m.tif",
    tree_vector_path="data/trees.geojson",
    grass_vector_path="data/grass.geojson",
    output_dir="results/optimization",
    sample_km_side=10,
    max_workers=8
)

# OPTION 2: 
center_coords = (679579, 2736332)  # Example UTM coordinates
print(f"Optimizing GLCM window size around coordinates: {center_coords}")
results_df, optimal_ws = optimize_window_size_from_vectors(
    panchromatic_path=panchromatic_path,
    tree_vector_path=tree_vector_path,
    grass_vector_path=grass_vector_path,
    output_dir=output_dir,
    sample_km_side=10,
    center_coords=center_coords,  
    max_workers=8
)
    
# 3. Process entire dataset
print(f"Running full GLCM analysis with {optimal_ws}x{optimal_ws} window...")
hybrid_parallel_glcm(
    input_path="data/riyadh_pan_0.35m.tif",
    output_dir="results/glcm_textures",
    window_size=optimal_ws,
    chunk_size=2048,
    max_workers=12
)

print("Workflow complete! Use textures for Random Forest classification.")
```

# Core Functions
```bash
optimize_window_size_from_vectors(panchromatic_path, tree_vector_path, grass_vector_path, output_dir, sample_km_size=10, center_coords=None, max_workers=8)
```
Find optimal GLCM window size using vector polygon training data.

Parameters:
- `panchromatic_path`: Path to panchromatic raster
- `tree_vector_path`: Path to tree polygon vector file
- `grass_vector_path`: Path to grass polygon vector file
- `output_dir`: Directory for temporary masks and results
- `sample_km_side`: Side length of square sample area in km
- `center_coords`: Optional (x,y) coordinates for specific sampling area
- `max_workers`: Number of parallel workers

```bash
hybrid_parallel_glcm(input_path, output_dir, window_size=11, chunk_size=2048, max_workers=None, metrics=None)
```
Compute GLCM textures in parallel across large raster datasets.

Parameters:
- `input_path`: Path to input raster (panchromatic recommended)
- `output_dir`: Directory to save GLCM texture results
- `window_size`: GLCM window size (must be odd)
- `chunk_size`: Processing chunk size in pixels (default: 2048)
- `max_workers`: Number of parallel workers (default: CPU count)
- `metrics`: GLCM metrics to compute: ['contrast', 'entropy', 'correlation']

# Input Data Requirements

## Raster Data

- Format: GeoTIFF recommended
- Resolution: High-resolution (0.3-1.0m optimal for texture analysis)
- Bands: Panchromatic or single-band for texture computation

## Vector Data

- Formats: GeoJSON, Shapefile, GPKG, GML
- Projection: Should match raster CRS (auto-reprojection available)
- Content: Polygon features for each class (trees, grass, etc.)

# Output Files

The package generates:

- `glcm_contrast.tif` - Texture contrast values
- `glcm_entropy.tif` - Texture entropy values
- `glcm_correlation.tif` - Texture correlation values
- `window_optimization_results.csv` - Optimization metrics
- Visualization plots for window size selection

Happy analyzing! 🌳🏙️📊