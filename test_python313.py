#!/usr/bin/env python3.13
"""
Test script to verify Python 3.13 compatibility while maintaining compatibility with earlier versions (3.7+).
"""

import sys
print(f"Python version: {sys.version}")

try:
    # Use the CORRECT spelling for scikit-image 0.25.2
    from skimage.feature import graycomatrix, graycoprops
    print("✓ graycomatrix/graycoprops imported successfully (American spelling)")
    
    import glcm_analyzer
    print("✓ glcm_analyzer imported successfully")
    
    from glcm_analyzer.core import hybrid_parallel_glcm
    from glcm_analyzer.optimization import optimize_window_size_from_vectors
    print("✓ All main functions imported successfully")
    
    import rasterio
    import geopandas as gpd
    import numpy as np
    
    print("✓ All dependencies imported successfully")
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} compatibility verified!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)