#!/usr/bin/env python3.13
"""
Diagnose scikit-image installation and available GLCM functions
"""

import sys
print(f"Python version: {sys.version}")

try:
    import skimage
    print(f"scikit-image version: {skimage.__version__}")
    
    # Check what's in skimage.feature
    from skimage.feature import __all__ as feature_all
    print(f"\nAvailable in skimage.feature: {feature_all}")
    
    # Try different import locations
    print("\nTrying different import paths:")
    
    # Method 1: New location (scikit-image >= 0.19)
    try:
        from skimage.feature import graycomatrix, graycoprops
        print("✓ Found graycomatrix/graycoprops in skimage.feature")
    except ImportError as e:
        print(f"✗ graycomatrix not in skimage.feature: {e}")
    
    # Method 2: Texture module (some versions)
    try:
        from skimage.feature.texture import graycomatrix, graycoprops
        print("✓ Found graycomatrix/graycoprops in skimage.feature.texture")
    except ImportError as e:
        print(f"✗ graycomatrix not in skimage.feature.texture: {e}")
    
    # Method 3: Direct texture import
    try:
        import skimage.feature.texture as texture
        print(f"✓ skimage.feature.texture available: {dir(texture)}")
    except ImportError as e:
        print(f"✗ skimage.feature.texture not available: {e}")
    
    # Method 4: Check if it exists in the main namespace but with different name
    try:
        from skimage.feature import greycomatrix, greycoprops
        print("✓ Found greycomatrix/greycoprops (British spelling)")
    except ImportError as e:
        print(f"✗ greycomatrix not found: {e}")
        
except ImportError as e:
    print(f"✗ Could not import scikit-image: {e}")