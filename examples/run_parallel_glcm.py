#!/usr/bin/env python3
"""
Example: Run parallel GLCM computation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from glcm_analyzer import hybrid_parallel_glcm

def main():
    input_image = "data/riyadh_pan_035m.tif"
    output_directory = "results/glcm_textures"
    
    print("Running parallel GLCM computation...")
    hybrid_parallel_glcm(
        input_path=input_image,
        output_dir=output_directory,
        window_size=11,  # Use optimal size from optimization
        chunk_size=2048,
        max_workers=12
    )

if __name__ == "__main__":
    main()