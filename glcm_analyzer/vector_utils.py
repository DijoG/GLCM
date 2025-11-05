import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import os
from rasterio.features import rasterize
from shapely.geometry import mapping
import pandas as pd


def vector_to_raster_mask(vector_path, raster_path, output_path, class_value=1, all_touched=True):
    """
    Convert vector polygons to raster mask aligned with reference raster
    
    Parameters:
    - vector_path: Path to vector file (GeoJSON, Shapefile, etc.)
    - raster_path: Path to reference raster for georeferencing
    - output_path: Path to save output raster mask
    - class_value: Value to assign to masked pixels
    - all_touched: Include all pixels touched by polygons (True) or only those whose center is within (False)
    """
    
    # Read vector data
    gdf = gpd.read_file(vector_path)
    print(f"Loaded {len(gdf)} features from {vector_path}")
    
    with rasterio.open(raster_path) as src:
        profile = src.profile.copy()
        profile.update({
            'dtype': rasterio.uint8,
            'count': 1,
            'compress': 'lzw',
            'nodata': 0
        })
        
        # Create mask array
        mask = rasterize(
            [(geom, class_value) for geom in gdf.geometry],
            out_shape=src.shape,
            transform=src.transform,
            fill=0,
            dtype=rasterio.uint8,
            all_touched=all_touched
        )
        
        # Save mask
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(mask, 1)
    
    print(f"Raster mask saved: {output_path}")
    print(f"Mask statistics: {np.sum(mask > 0):,} pixels masked")
    return output_path


def extract_samples_from_vector(vector_path, raster_path, sample_fraction=1.0, random_seed=42):
    """
    Extract sample coordinates from vector polygons
    
    Parameters:
    - vector_path: Path to vector file
    - raster_path: Path to reference raster
    - sample_fraction: Fraction of pixels to sample (0.0-1.0)
    - random_seed: Random seed for reproducible sampling
    """
    
    gdf = gpd.read_file(vector_path)
    
    with rasterio.open(raster_path) as src:
        all_coords = []
        
        for idx, geom in enumerate(gdf.geometry):
            # Rasterize individual geometry
            mask = rasterize(
                [(geom, 1)],
                out_shape=src.shape,
                transform=src.transform,
                fill=0,
                dtype=rasterio.uint8
            )
            
            # Get coordinates of masked pixels
            rows, cols = np.where(mask == 1)
            if len(rows) > 0:
                # Convert to geographic coordinates
                xs, ys = rasterio.transform.xy(src.transform, rows, cols)
                coords = list(zip(xs, ys))
                
                # Sample if needed
                if sample_fraction < 1.0:
                    np.random.seed(random_seed + idx)
                    n_samples = int(len(coords) * sample_fraction)
                    if n_samples > 0:
                        sampled_coords = coords[np.random.choice(len(coords), n_samples, replace=False)]
                        all_coords.extend(sampled_coords)
                else:
                    all_coords.extend(coords)
        
        print(f"Extracted {len(all_coords)} sample coordinates from {len(gdf)} polygons")
        return all_coords


def create_training_masks_from_vectors(tree_vector_path, grass_vector_path, raster_path, output_dir, 
                                     sample_fraction=0.1, buffer_size=0):
    """
    Create training masks from vector polygons for tree and grass classes
    
    Parameters:
    - tree_vector_path: Path to tree polygon vector file
    - grass_vector_path: Path to grass polygon vector file  
    - raster_path: Path to reference raster
    - output_dir: Directory to save output masks
    - sample_fraction: Fraction of pixels to include (for large polygons)
    - buffer_size: Buffer size in meters (optional, for point data)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process tree polygons
    print("Processing tree polygons...")
    tree_mask_path = os.path.join(output_dir, 'tree_samples.tif')
    vector_to_raster_mask(tree_vector_path, raster_path, tree_mask_path, class_value=1)
    
    # Process grass polygons  
    print("Processing grass polygons...")
    grass_mask_path = os.path.join(output_dir, 'grass_samples.tif')
    vector_to_raster_mask(grass_vector_path, raster_path, grass_mask_path, class_value=1)
    
    # Validate masks
    with rasterio.open(tree_mask_path) as src:
        tree_mask = src.read(1)
    with rasterio.open(grass_mask_path) as src:
        grass_mask = src.read(1)
    
    tree_pixels = np.sum(tree_mask > 0)
    grass_pixels = np.sum(grass_mask > 0)
    
    print(f"\nTraining mask statistics:")
    print(f"Tree samples: {tree_pixels:,} pixels")
    print(f"Grass samples: {grass_pixels:,} pixels")
    print(f"Total samples: {tree_pixels + grass_pixels:,} pixels")
    
    # Check for overlap
    overlap = np.sum((tree_mask > 0) & (grass_mask > 0))
    if overlap > 0:
        print(f"Warning: {overlap:,} overlapping pixels between classes!")
    
    return tree_mask_path, grass_mask_path


def validate_vector_raster_alignment(vector_path, raster_path):
    """
    Validate that vector and raster are in the same CRS and extent
    """
    gdf = gpd.read_file(vector_path)
    
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
    
    # Check CRS
    if gdf.crs != raster_crs:
        print(f"CRS mismatch: Vector={gdf.crs}, Raster={raster_crs}")
        print("Reprojecting vector to raster CRS...")
        gdf = gdf.to_crs(raster_crs)
    
    # Check extent
    vector_bounds = gdf.total_bounds
    print(f"Raster bounds: {raster_bounds}")
    print(f"Vector bounds: {vector_bounds}")
    
    # Check if vector is within raster
    if (vector_bounds[0] < raster_bounds[0] or vector_bounds[2] > raster_bounds[2] or
        vector_bounds[1] < raster_bounds[1] or vector_bounds[3] > raster_bounds[3]):
        print("Warning: Vector extends beyond raster bounds")
    
    return gdf


def get_vector_statistics(vector_path):
    """
    Get basic statistics about vector data
    """
    gdf = gpd.read_file(vector_path)
    
    stats = {
        'num_features': len(gdf),
        'crs': gdf.crs,
        'bounds': gdf.total_bounds,
        'geometry_types': gdf.geometry.type.unique().tolist(),
        'total_area_km2': gdf.geometry.area.sum() / 1e6,
        'mean_area_km2': gdf.geometry.area.mean() / 1e6
    }
    
    print(f"Vector Statistics for {os.path.basename(vector_path)}:")
    print(f"  Number of features: {stats['num_features']}")
    print(f"  CRS: {stats['crs']}")
    print(f"  Geometry types: {stats['geometry_types']}")
    print(f"  Total area: {stats['total_area_km2']:.2f} km²")
    print(f"  Mean feature area: {stats['mean_area_km2']:.4f} km²")
    
    return stats