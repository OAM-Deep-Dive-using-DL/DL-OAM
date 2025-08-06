#!/usr/bin/env python3
import os
import argparse
import numpy as np
from utils.visualization_unified import (
    plot_oam_mode,
    plot_multiple_oam_modes,
    plot_oam_mode_3d,
    visualize_oam_propagation,
    plot_oam_superposition,
    visualize_oam_impairments,
    visualize_oam_mode_coupling,
    visualize_oam_crosstalk_matrix
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate OAM mode visualizations")
    
    parser.add_argument("--output-dir", type=str, default="plots/oam_modes",
                       help="Directory to save visualizations")
    parser.add_argument("--mode-range", type=int, nargs=2, default=[1, 6],
                       help="Range of OAM modes to visualize (min max)")
    parser.add_argument("--size", type=int, default=500,
                       help="Size of the grid in pixels")
    parser.add_argument("--beam-width", type=float, default=0.3,
                       help="Relative beam width")
    parser.add_argument("--show", action="store_true",
                       help="Show plots instead of saving them")
    parser.add_argument("--specific-mode", type=int, default=None,
                       help="Visualize only a specific OAM mode")
    parser.add_argument("--3d", dest="plot_3d", action="store_true",
                       help="Generate 3D visualizations")
    
    return parser.parse_args()


def main():
    """Main function to generate OAM visualizations."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set mode range
    min_mode, max_mode = args.mode_range
    if args.specific_mode is not None:
        modes = [args.specific_mode]
    else:
        modes = list(range(min_mode, max_mode + 1))
    
    print(f"Generating OAM mode visualizations for modes {modes}...")
    
    # Generate individual mode plots
    for l in modes:
        print(f"Generating visualization for OAM mode {l}...")
        save_path = None if args.show else os.path.join(args.output_dir, f"oam_mode_{l}.png")
        plot_oam_mode(l, size=args.size, beam_width=args.beam_width, 
                     save_path=save_path, show=args.show)
    
    # Generate comparison of multiple modes
    if len(modes) > 1:
        print("Generating OAM modes comparison...")
        save_dir = None if args.show else args.output_dir
        plot_multiple_oam_modes(modes, size=args.size, beam_width=args.beam_width, 
                              save_dir=save_dir, show=args.show)
    
    # Generate 3D visualizations
    if args.plot_3d:
        print("Generating 3D visualizations...")
        for l in modes:
            save_path = None if args.show else os.path.join(args.output_dir, f"oam_mode_{l}_3d.png")
            plot_oam_mode_3d(l, size=min(args.size, 200), beam_width=args.beam_width, 
                           save_path=save_path, show=args.show)
    
    # Generate propagation visualization
    print("Generating OAM propagation visualization...")
    for l in modes:
        save_path = None if args.show else os.path.join(args.output_dir, f"oam_mode_{l}_propagation.png")
        visualize_oam_propagation(l, [0, 1, 2, 3, 4], size=args.size, beam_width=args.beam_width, 
                                save_path=save_path, show=args.show)
    
    # Generate OAM superposition visualizations
    if len(modes) >= 2:
        print("Generating OAM superposition visualizations...")
        # Pairs of modes
        for i in range(len(modes) - 1):
            l1, l2 = modes[i], modes[i+1]
            save_path = None if args.show else os.path.join(args.output_dir, f"oam_superposition_{l1}_{l2}.png")
            plot_oam_superposition([l1, l2], [1.0, 0.5], size=args.size, beam_width=args.beam_width,
                                 save_path=save_path, show=args.show)
        
        # Triple mode superposition if we have enough modes
        if len(modes) >= 3:
            l1, l2, l3 = modes[0], modes[len(modes)//2], modes[-1]
            save_path = None if args.show else os.path.join(args.output_dir, f"oam_superposition_{l1}_{l2}_{l3}.png")
            plot_oam_superposition([l1, l2, l3], [0.7, 0.5, 0.3], size=args.size, beam_width=args.beam_width,
                                 save_path=save_path, show=args.show)
    
    # Generate OAM impairments visualizations
    print("Generating OAM impairments visualizations...")
    for l in modes:
        save_dir = None if args.show else args.output_dir
        visualize_oam_impairments(l, size=args.size, beam_width=args.beam_width,
                                save_dir=save_dir, show=args.show)
    
    # Generate OAM mode coupling visualizations
    if len(modes) >= 2:
        print("Generating OAM mode coupling visualizations...")
        for i in range(len(modes) - 1):
            l1, l2 = modes[i], modes[i+1]
            save_path = None if args.show else os.path.join(args.output_dir, f"oam_coupling_{l1}_{l2}.png")
            visualize_oam_mode_coupling(l1, l2, 0.3, size=args.size, beam_width=args.beam_width,
                                      save_path=save_path, show=args.show)
    
    # Generate OAM crosstalk matrix visualization
    print("Generating OAM crosstalk matrix visualization...")
    save_path = None if args.show else os.path.join(args.output_dir, "oam_crosstalk_matrix.png")
    visualize_oam_crosstalk_matrix(max(modes), 0.5, save_path=save_path, show=args.show)
    
    if not args.show:
        print(f"All OAM mode visualizations have been generated in the {args.output_dir} directory.")


if __name__ == "__main__":
    main() 