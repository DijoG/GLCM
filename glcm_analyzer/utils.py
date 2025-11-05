import os
import rasterio
import numpy as np
import geopandas as gpd


def validate_inputs(*paths):
    """Validate that input files exist"""
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")


def create_output_profile(input_profile):
    """Create output profile for GLCM results"""
    profile = input_profile.copy()
    profile.update({
        'dtype': rasterio.float32,
        'count': 1,
        'compress': 'lzw',
        'nodata': 0
    })
    return profile


def create_sample_masks(panchromatic_path, output_dir, tree_coords, grass_coords, buffer_size=10):
    """
    Create sample masks from coordinate lists
    
    Parameters:
    - panchromatic_path: Path to panchromatic raster for georeferencing
    - output_dir: Directory to save masks
    - tree_coords: List of (x, y) tree coordinates
    - grass_coords: List of (x, y) grass coordinates  
    - buffer_size: Buffer size around points in pixels
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with rasterio.open(panchromatic_path) as src:
        profile = src.profile.copy()
        profile.update({'count': 1, 'dtype': rasterio.uint8})
        
        tree_mask = np.zeros(src.shape, dtype=np.uint8)
        grass_mask = np.zeros(src.shape, dtype=np.uint8)
        
        # Create tree mask
        for x, y in tree_coords:
            row, col = src.index(x, y)
            r_start = max(0, row - buffer_size)
            r_end = min(src.height, row + buffer_size + 1)
            c_start = max(0, col - buffer_size)
            c_end = min(src.width, col + buffer_size + 1)
            tree_mask[r_start:r_end, c_start:c_end] = 1
        
        # Create grass mask
        for x, y in grass_coords:
            row, col = src.index(x, y)
            r_start = max(0, row - buffer_size)
            r_end = min(src.height, row + buffer_size + 1)
            c_start = max(0, col - buffer_size)
            c_end = min(src.width, col + buffer_size + 1)
            grass_mask[r_start:r_end, c_start:c_end] = 1
        
        # Save masks
        tree_path = os.path.join(output_dir, 'tree_samples.tif')
        grass_path = os.path.join(output_dir, 'grass_samples.tif')
        
        with rasterio.open(tree_path, 'w', **profile) as dst:
            dst.write(tree_mask, 1)
        
        with rasterio.open(grass_path, 'w', **profile) as dst:
            dst.write(grass_mask, 1)
        
        print(f"Tree mask saved: {tree_path}")
        print(f"Grass mask saved: {grass_path}")
        
    return tree_path, grass_path

# Add to existing utils.py
def validate_vector_file(vector_path):
    """Validate that vector file exists and can be read"""
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Vector file not found: {vector_path}")
    
    try:
        gdf = gpd.read_file(vector_path)
        if len(gdf) == 0:
            raise ValueError(f"Vector file is empty: {vector_path}")
        return gdf
    except Exception as e:
        raise ValueError(f"Error reading vector file {vector_path}: {e}")


def check_crs_compatibility(vector_path, raster_path):
    """Check if vector and raster have compatible CRS"""
    gdf = gpd.read_file(vector_path)
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
    
    if gdf.crs != raster_crs:
        print(f"CRS mismatch: Vector={gdf.crs}, Raster={raster_crs}")
        return False
    return True