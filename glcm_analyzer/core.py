import rasterio
import numpy as np
from skimage.feature import graycomatrix, graycoprops  

from rasterio.windows import Window
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import os
from .utils import validate_inputs, create_output_profile


def optimized_glcm_window(data, window_size=11, metric='contrast'):
    """Highly optimized GLCM for single window"""
    if data.std() < 5:  # Skip homogeneous areas (adjust threshold)
        return 0
    
    # Fast quantization
    data_min, data_max = data.min(), data.max()
    if data_max - data_min < 10:  # Low dynamic range
        return 0
        
    data_uint8 = ((data - data_min) / (data_max - data_min + 1e-8) * 255).astype(np.uint8)
    
    try:
        glcm = graycomatrix(data_uint8, [1], [0], 256, symmetric=True, normed=True)
        return graycoprops(glcm, metric)[0, 0]
    except:
        return 0


def process_tile_optimized(tile_info):
    """Process a single tile with multiple textures"""
    j, i, chunk_size, input_path, window_size = tile_info
    
    try:
        with rasterio.open(input_path) as src:
            win = Window(j, i, 
                        min(chunk_size, src.width - j), 
                        min(chunk_size, src.height - i))
            chunk_data = src.read(1, window=win)
            
            if chunk_data.size == 0:
                return j, i, None, None, None
                
            height, width = chunk_data.shape
            pad = window_size // 2
            
            # Pad the chunk
            chunk_padded = np.pad(chunk_data, pad, mode='reflect')
            
            # Pre-allocate output arrays
            contrast = np.zeros((height, width), dtype=np.float32)
            entropy = np.zeros((height, width), dtype=np.float32)
            correlation = np.zeros((height, width), dtype=np.float32)
            
            # Process with sliding window
            for row in range(height):
                for col in range(width):
                    window = chunk_padded[row:row+window_size, col:col+window_size]
                    # Compute multiple textures in one pass
                    if window.std() > 5:
                        window_uint8 = ((window - window.min()) / 
                                      (window.max() - window.min() + 1e-8) * 255).astype(np.uint8)
                        glcm = graycomatrix(window_uint8, [1], [0], 256, symmetric=True, normed=True)
                        contrast[row, col] = graycoprops(glcm, 'contrast')[0, 0]
                        entropy[row, col] = graycoprops(glcm, 'entropy')[0, 0]
                        correlation[row, col] = graycoprops(glcm, 'correlation')[0, 0]
            
            return j, i, contrast, entropy, correlation
    except Exception as e:
        print(f"Error processing tile ({j}, {i}): {e}")
        return j, i, None, None, None


def hybrid_parallel_glcm(input_path, output_dir, window_size=11, 
                        chunk_size=2048, max_workers=None, metrics=None):
    """
    Hybrid parallel processing optimized for large areas
    
    Parameters:
    - input_path: Path to input raster (panchromatic recommended)
    - output_dir: Directory to save GLCM texture results
    - window_size: Size of GLCM window (must be odd)
    - chunk_size: Size of processing chunks in pixels
    - max_workers: Number of parallel workers
    - metrics: List of GLCM metrics to compute ['contrast', 'entropy', 'correlation']
    """
    
    if metrics is None:
        metrics = ['contrast', 'entropy', 'correlation']
    
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
    
    validate_inputs(input_path, output_dir)
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 16)  # Cap at 16 to avoid thrashing
    
    os.makedirs(output_dir, exist_ok=True)
    
    with rasterio.open(input_path) as src:
        profile = create_output_profile(src.profile)
        
        # Generate tile coordinates
        tiles = []
        for i in range(0, src.height, chunk_size):
            for j in range(0, src.width, chunk_size):
                tiles.append((j, i, chunk_size, input_path, window_size))
        
        print(f"Processing {len(tiles)} tiles with {max_workers} workers...")
        print(f"Input: {input_path}")
        print(f"Output: {output_dir}")
        print(f"Window size: {window_size}x{window_size}")
        print(f"Chunk size: {chunk_size}x{chunk_size}")
        
        # Process tiles in parallel with progress tracking
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_tile = {executor.submit(process_tile_optimized, tile): tile 
                            for tile in tiles}
            
            for future in tqdm(as_completed(future_to_tile), total=len(tiles), 
                             desc="Processing tiles"):
                results.append(future.result())
        
        # Write results
        output_paths = {}
        writers = {}
        
        for metric in metrics:
            output_path = os.path.join(output_dir, f'glcm_{metric}.tif')
            output_paths[metric] = output_path
            writers[metric] = rasterio.open(output_path, 'w', **profile)
        
        for j, i, contrast_chunk, entropy_chunk, correlation_chunk in results:
            if contrast_chunk is not None:
                win = Window(j, i, contrast_chunk.shape[1], contrast_chunk.shape[0])
                
                if 'contrast' in metrics:
                    writers['contrast'].write(contrast_chunk, 1, window=win)
                if 'entropy' in metrics:
                    writers['entropy'].write(entropy_chunk, 1, window=win) 
                if 'correlation' in metrics:
                    writers['correlation'].write(correlation_chunk, 1, window=win)
        
        for writer in writers.values():
            writer.close()
        
        print("GLCM computation complete!")
        print("Output files:")
        for metric, path in output_paths.items():
            print(f"  - {path}")


def calculate_texture_strided(data, window_size, metric='contrast'):
    """Calculate texture using strided windows for efficiency"""
    from numpy.lib.stride_tricks import sliding_window_view
    
    pad = window_size // 2
    data_padded = np.pad(data, pad, mode='reflect')
    
    windows = sliding_window_view(data_padded, (window_size, window_size))
    height, width = windows.shape[0], windows.shape[1]
    
    def compute_single_window(window):
        if window.std() < 5:
            return 0
        window_uint8 = ((window - window.min()) / 
                       (window.max() - window.min() + 1e-8) * 255).astype(np.uint8)
        glcm = graycomatrix(window_uint8, [1], [0], 256, symmetric=True, normed=True)
        return graycoprops(glcm, metric)[0, 0]
    
    texture_flat = np.array([compute_single_window(w) for w in 
                           windows.reshape(-1, window_size, window_size)])
    return texture_flat.reshape(height, width)