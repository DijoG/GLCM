import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.feature import graycomatrix, graycoprops
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from .core import calculate_texture_strided
from .utils import validate_inputs
import os


def evaluate_window_size_performance(data, window_size, class_mask_tree, class_mask_grass):
    """
    Evaluate GLCM performance for a specific window size
    """
    height, width = data.shape
    pad = window_size // 2
    data_padded = np.pad(data, pad, mode='reflect')
    
    # Initialize texture arrays
    contrast = np.zeros((height, width), dtype=np.float32)
    
    # Calculate texture
    for i in range(height):
        for j in range(width):
            window = data_padded[i:i+window_size, j:j+window_size]
            if window.std() > 5:  # Skip homogeneous areas
                window_uint8 = ((window - window.min()) / 
                              (window.max() - window.min() + 1e-8) * 255).astype(np.uint8)
                glcm = graycomatrix(window_uint8, [1], [0], 256, symmetric=True, normed=True)
                contrast[i, j] = graycoprops(glcm, 'contrast')[0, 0]
    
    # Calculate class separation metrics
    tree_values = contrast[class_mask_tree]
    grass_values = contrast[class_mask_grass]
    
    # Remove zeros and NaNs
    tree_values = tree_values[tree_values > 0]
    grass_values = grass_values[grass_values > 0]
    
    if len(tree_values) == 0 or len(grass_values) == 0:
        return window_size, 0, 0, 0, 0
    
    # Calculate separation metrics
    mean_tree = np.mean(tree_values)
    mean_grass = np.mean(grass_values)
    std_tree = np.std(tree_values)
    std_grass = np.std(grass_values)
    
    # Separation score (higher is better)
    separation_score = abs(mean_tree - mean_grass) / (std_tree + std_grass + 1e-8)
    
    # Overlap percentage (lower is better)
    threshold = (mean_tree + mean_grass) / 2
    overlap_pct = (np.sum(grass_values > threshold) + np.sum(tree_values < threshold)) / \
                  (len(tree_values) + len(grass_values))
    
    return window_size, separation_score, overlap_pct, mean_tree, mean_grass


def optimize_window_size(panchromatic_path, tree_mask_path, grass_mask_path, 
                        sample_km_side=10, center_coords=None, max_workers=8):
    """
    Find optimal GLCM window size by testing on a sample area
    
    Parameters:
    - panchromatic_path: Path to panchromatic band
    - tree_mask_path: Path to binary tree mask (1=trees, 0=non-trees)
    - grass_mask_path: Path to binary grass/low vegetation mask
    - sample_km_side: Size of sample area in km (default 10x10 km)
    - center_coords: Optional (x, y) center coordinates for sampling
    - max_workers: Number of parallel workers for testing
    """
    
    validate_inputs(panchromatic_path)
    validate_inputs(tree_mask_path)
    validate_inputs(grass_mask_path)
    
    with rasterio.open(panchromatic_path) as src:
        # Calculate sample area in pixels
        resolution = src.res[0]  # assuming square pixels
        sample_pixels = int((sample_km_side * 1000) / resolution)
        print(f"Sample area: {sample_pixels}x{sample_pixels} pixels "
              f"({sample_pixels * resolution / 1000:.1f}x{sample_pixels * resolution / 1000:.1f} km)")
        
        # Determine sample window
        if center_coords:
            x_center, y_center = center_coords
        else:
            # Sample from center of image
            x_center, y_center = src.width // 2, src.height // 2
        
        x_start = max(0, x_center - sample_pixels // 2)
        y_start = max(0, y_center - sample_pixels // 2)
        x_end = min(src.width, x_start + sample_pixels)
        y_end = min(src.height, y_start + sample_pixels)
        
        sample_window = Window(x_start, y_start, x_end - x_start, y_end - y_start)
        
        # Read sample data
        pan_sample = src.read(1, window=sample_window)
        print(f"Sample data shape: {pan_sample.shape}")
    
    # Read classification masks
    with rasterio.open(tree_mask_path) as src:
        tree_mask = src.read(1, window=sample_window).astype(bool)
    
    with rasterio.open(grass_mask_path) as src:
        grass_mask = src.read(1, window=sample_window).astype(bool)
    
    print(f"Tree pixels in sample: {np.sum(tree_mask):,}")
    print(f"Grass pixels in sample: {np.sum(grass_mask):,}")
    
    # Test window sizes (odd numbers only)
    window_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 31, 35, 41]
    
    # Evaluate window sizes in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for ws in window_sizes:
            future = executor.submit(
                evaluate_window_size_performance, 
                pan_sample, ws, tree_mask, grass_mask
            )
            futures.append(future)
        
        for future in tqdm(futures, desc="Testing window sizes"):
            results.append(future.result())
    
    # Analyze results
    df_results = pd.DataFrame(results, columns=[
        'window_size', 'separation_score', 'overlap_pct', 'mean_tree', 'mean_grass'
    ])
    
    # Find optimal window size
    optimal_idx = df_results['separation_score'].idxmax()
    optimal_ws = df_results.loc[optimal_idx, 'window_size']
    optimal_separation = df_results.loc[optimal_idx, 'separation_score']
    
    print(f"\n{'='*50}")
    print(f"OPTIMAL WINDOW SIZE: {optimal_ws}x{optimal_ws}")
    print(f"Separation Score: {optimal_separation:.3f}")
    print(f"Class Overlap: {df_results.loc[optimal_idx, 'overlap_pct']:.1%}")
    print(f"Tree Mean Contrast: {df_results.loc[optimal_idx, 'mean_tree']:.1f}")
    print(f"Grass Mean Contrast: {df_results.loc[optimal_idx, 'mean_grass']:.1f}")
    print(f"{'='*50}")
    
    # Plot results
    plot_window_size_results(df_results, optimal_ws)
    
    return df_results, optimal_ws


def plot_window_size_results(df_results, optimal_ws):
    """Plot the window size optimization results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot separation score
    ax1.plot(df_results['window_size'], df_results['separation_score'], 
             'bo-', linewidth=2, markersize=8, label='Separation Score')
    ax1.axvline(optimal_ws, color='red', linestyle='--', 
                label=f'Optimal: {optimal_ws}x{optimal_ws}')
    ax1.set_xlabel('Window Size (pixels)')
    ax1.set_ylabel('Separation Score (higher = better)')
    ax1.set_title('GLCM Window Size vs Class Separation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot class means and overlap
    ax2.plot(df_results['window_size'], df_results['mean_tree'], 
             'g^-', linewidth=2, label='Tree Contrast')
    ax2.plot(df_results['window_size'], df_results['mean_grass'], 
             'bs-', linewidth=2, label='Grass Contrast')
    ax2.set_xlabel('Window Size (pixels)')
    ax2.set_ylabel('GLCM Contrast Value')
    ax2.set_title('Texture Values by Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('window_size_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()


def quick_visual_assessment(panchromatic_path, window_sizes=[7, 11, 15, 21]):
    """
    Quickly visualize different window sizes on a small sample
    """
    with rasterio.open(panchromatic_path) as src:
        # Take a small sample for quick visualization
        sample_size = 1000
        sample_window = Window(0, 0, sample_size, sample_size)
        pan_sample = src.read(1, window=sample_window)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, ws in enumerate(window_sizes):
        # Calculate contrast for this window size
        contrast = calculate_texture_strided(pan_sample, ws)
        
        # Display
        im = axes[idx].imshow(contrast, cmap='viridis', vmax=np.percentile(contrast, 95))
        axes[idx].set_title(f'Window Size: {ws}x{ws}\nMax Contrast: {contrast.max():.1f}')
        plt.colorbar(im, ax=axes[idx])
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('window_size_visual_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Add to existing optimization.py
def optimize_window_size_from_vectors(panchromatic_path, tree_vector_path, grass_vector_path,
                                    output_dir, sample_km_side=10, center_coords=None, max_workers=8):
    """
    Find optimal GLCM window size using vector polygon inputs
    
    Parameters:
    - panchromatic_path: Path to panchromatic raster
    - tree_vector_path: Path to tree polygon vector file
    - grass_vector_path: Path to grass polygon vector file
    - output_dir: Directory to save temporary masks and results
    - sample_km_side: Side length of sample area in km
    - center_coords: Optional center coordinates for sampling
    - max_workers: Number of parallel workers
    """
    
    from .vector_utils import create_training_masks_from_vectors
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating training masks from vector polygons...")
    tree_mask_path, grass_mask_path = create_training_masks_from_vectors(
        tree_vector_path=tree_vector_path,
        grass_vector_path=grass_vector_path,
        raster_path=panchromatic_path,
        output_dir=output_dir
    )
    
    print("Optimizing window size...")
    results_df, optimal_ws = optimize_window_size(
        panchromatic_path=panchromatic_path,
        tree_mask_path=tree_mask_path,
        grass_mask_path=grass_mask_path,
        sample_km_side=sample_km_side,
        center_coords=center_coords,
        max_workers=max_workers
    )
    
    return results_df, optimal_ws