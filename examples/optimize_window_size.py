#!/usr/bin/env python3
"""
Example: Optimize GLCM window size using vector polygon inputs
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from glcm_analyzer import optimize_window_size_from_vectors, get_vector_statistics

def main():
    # Input files
    panchromatic_path = "data/riyadh_pan_035m.tif"
    tree_vector_path = "data/training/tree_polygons.geojson"  
    grass_vector_path = "data/training/grass_polygons.geojson"
    output_dir = "results/window_optimization"
    
    print("=== GLCM Window Size Optimization with Vector Inputs ===")
    
    # Check vector statistics
    print("\n1. Analyzing input vectors...")
    tree_stats = get_vector_statistics(tree_vector_path)
    grass_stats = get_vector_statistics(grass_vector_path)
    
    print("\n2. Optimizing GLCM window size...")
    
    # OPTION 1: Use default center (middle of image)
    results_df, optimal_ws = optimize_window_size_from_vectors(
        panchromatic_path=panchromatic_path,
        tree_vector_path=tree_vector_path,
        grass_vector_path=grass_vector_path,
        output_dir=output_dir,
        sample_km_side=10,      # 10x10 km sample area
        max_workers=8
        # center_coords not specified - uses default None (center of image)
    )
    
    # OPTION 2: If you want to specify a specific area, uncomment and modify:
    # center_coords = (679579, 2736332)  # Example UTM coordinates
    # results_df, optimal_ws = optimize_window_size_from_vectors(
    #     panchromatic_path=panchromatic_path,
    #     tree_vector_path=tree_vector_path,
    #     grass_vector_path=grass_vector_path,
    #     output_dir=output_dir,
    #     sample_km_side=10,
    #     center_coords=center_coords,  
    #     max_workers=8
    # )
    
    print(f"\n3. RESULTS:")
    print(f"   Optimal window size: {optimal_ws}x{optimal_ws} pixels")
    print(f"   Best separation score: {results_df['separation_score'].max():.3f}")
    
    # Save detailed results
    results_csv = os.path.join(output_dir, "window_optimization_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"   Detailed results saved: {results_csv}")

if __name__ == "__main__":
    main()