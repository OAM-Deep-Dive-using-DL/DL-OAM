#!/usr/bin/env python3
"""
Plot Organization Utility

This script helps organize and maintain the plots directory structure.
All visualization outputs from the OAM project are centralized in the plots/ directory.
"""

import os
import shutil
import glob
from pathlib import Path

def create_plots_directory():
    """Create the plots directory structure."""
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different types of plots
    subdirs = [
        "physics",      # Physics validation plots
        "training",     # RL training plots  
        "evaluation",   # Model evaluation plots
        "analysis",     # Performance analysis plots
        "comparison"    # Comparison studies
    ]
    
    for subdir in subdirs:
        (plots_dir / subdir).mkdir(exist_ok=True)
    
    print(f"✅ Created plots directory structure in {plots_dir.absolute()}")

def move_existing_plots():
    """Move any existing PNG files to the plots directory."""
    # Find PNG files in the root directory
    png_files = glob.glob("*.png")
    
    if not png_files:
        print("No PNG files found in root directory")
        return
    
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    moved_count = 0
    for png_file in png_files:
        dest_path = plots_dir / png_file
        try:
            shutil.move(png_file, dest_path)
            print(f"Moved {png_file} -> {dest_path}")
            moved_count += 1
        except Exception as e:
            print(f"❌ Error moving {png_file}: {e}")
    
    print(f"✅ Moved {moved_count} PNG files to plots directory")

def clean_plots_directory():
    """Clean up old or duplicate plots."""
    plots_dir = Path("plots")
    if not plots_dir.exists():
        print("Plots directory does not exist")
        return
    
    # Count existing plots
    png_files = list(plots_dir.glob("**/*.png"))
    print(f"📊 Found {len(png_files)} PNG files in plots directory:")
    
    for png_file in sorted(png_files):
        size_kb = png_file.stat().st_size / 1024
        print(f"  - {png_file.name}: {size_kb:.1f} KB")

def list_plots():
    """List all plots in the directory with metadata."""
    plots_dir = Path("plots")
    if not plots_dir.exists():
        print("Plots directory does not exist")
        return
    
    print(f"\n📊 Plot Inventory ({plots_dir.absolute()})")
    print("=" * 60)
    
    # Enhanced physics plots
    enhanced_plots = [
        "enhanced_phase_screen_fft.png",
        "enhanced_turbulence_spectra.png", 
        "enhanced_multi_layer_analysis.png",
        "enhanced_aperture_averaging.png",
        "enhanced_inner_outer_scale.png",
        "enhanced_full_channel_analysis.png"
    ]
    
    print("\n🚀 Enhanced Physics Plots:")
    for plot in enhanced_plots:
        plot_path = plots_dir / "physics" / "enhanced" / plot
        if plot_path.exists():
            size_kb = plot_path.stat().st_size / 1024
            print(f"  ✅ {plot}: {size_kb:.1f} KB")
        else:
            print(f"  ❌ {plot}: Missing")
    
    # Basic plots
    basic_plots = [
        "phase_screen_fft.png",
        "non_kolmogorov_psd.png",
        "cn2_profile.png", 
        "multi_layer_phase_screens.png",
        "aperture_averaging.png",
        "inner_outer_scale.png",
        "advanced_physics_sinr.png"
    ]
    
    print("\n📈 Basic Physics Plots:")
    for plot in basic_plots:
        plot_path = plots_dir / "physics" / "basic" / plot
        if plot_path.exists():
            size_kb = plot_path.stat().st_size / 1024
            print(f"  ✅ {plot}: {size_kb:.1f} KB")
        else:
            print(f"  ⚪ {plot}: Not generated")
    
    # Training/evaluation plots
    other_plots = []
    for subdir in ["training", "evaluation", "analysis", "comparison"]:
        if (plots_dir / subdir).exists():
            other_plots.extend(list((plots_dir / subdir).glob("*.png")))
    
    if other_plots:
        print(f"\n🔬 Other Plots:")
        for plot in sorted(other_plots):
            size_kb = plot.stat().st_size / 1024
            print(f"  📊 {plot.relative_to(plots_dir)}: {size_kb:.1f} KB")
    
    total_size = sum(p.stat().st_size for p in plots_dir.glob("**/*.png")) / 1024 / 1024
    total_count = len(list(plots_dir.glob("**/*.png")))
    print(f"\n📋 Summary: {total_count} plots, {total_size:.1f} MB total")

def main():
    """Main function with command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Organize plots directory for OAM project")
    parser.add_argument("--create", action="store_true", help="Create plots directory structure")
    parser.add_argument("--move", action="store_true", help="Move existing PNG files to plots/")
    parser.add_argument("--clean", action="store_true", help="Clean up plots directory")
    parser.add_argument("--list", action="store_true", help="List all plots with metadata")
    parser.add_argument("--all", action="store_true", help="Run all organization tasks")
    
    args = parser.parse_args()
    
    if args.all or args.create:
        create_plots_directory()
    
    if args.all or args.move:
        move_existing_plots()
    
    if args.all or args.clean:
        clean_plots_directory()
    
    if args.all or args.list or (not any([args.create, args.move, args.clean])):
        list_plots()

if __name__ == "__main__":
    main() 