"""
GLCM Analyzer - A high-performance GLCM texture analysis package for satellite imagery
"""

from .core import hybrid_parallel_glcm, calculate_texture_strided
from .optimization import (optimize_window_size, optimize_window_size_from_vectors, 
                          quick_visual_assessment, plot_window_size_results,
                          evaluate_window_size_performance)
from .vector_utils import (vector_to_raster_mask, extract_samples_from_vector,
                          create_training_masks_from_vectors, validate_vector_raster_alignment,
                          get_vector_statistics)
from .utils import (validate_inputs, create_output_profile, create_sample_masks,
                   validate_vector_file, check_crs_compatibility)

__version__ = "0.1.0"
__author__ = "Gergo Dioszegi"
__email__ = "dijogergo@gmail.com"

__all__ = [
    # Core functions
    'hybrid_parallel_glcm',
    'calculate_texture_strided', 
    
    # Optimization
    'optimize_window_size',
    'optimize_window_size_from_vectors',
    'quick_visual_assessment',
    'plot_window_size_results',
    'evaluate_window_size_performance',
    
    # Vector utilities
    'vector_to_raster_mask',
    'extract_samples_from_vector',
    'create_training_masks_from_vectors',
    'validate_vector_raster_alignment',
    'get_vector_statistics',
    
    # General utilities
    'validate_inputs',
    'create_output_profile',
    'create_sample_masks',
    'validate_vector_file',
    'check_crs_compatibility',
]