#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulator.channel_simulator import ChannelSimulator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def test_fft_phase_screen(simulator: ChannelSimulator) -> None:
    """Test FFT-based phase screen generation."""
    print("\n=== Testing FFT-based Phase Screen Generation ===")
    
    # Generate phase screen
    r0 = 0.1  # Fried parameter
    distance = 100.0  # meters
    
    # Generate phase screen
    phase_screen = simulator._generate_fft_phase_screen(r0, distance)
    
    print(f"Phase screen shape: {phase_screen.shape}")
    print(f"Phase screen min: {np.min(phase_screen):.4f}, max: {np.max(phase_screen):.4f}")
    print(f"Phase screen mean: {np.mean(phase_screen):.4f}, std: {np.std(phase_screen):.4f}")
    
    # Plot phase screen
    plt.figure(figsize=(10, 8))
    plt.imshow(phase_screen, cmap='jet')
    plt.colorbar(label='Phase (rad)')
    plt.title(f'FFT-based Phase Screen (r0={r0:.3f}m, distance={distance}m)')
    plt.tight_layout()
    os.makedirs('plots/physics/basic', exist_ok=True)
    plt.savefig('plots/physics/basic/phase_screen_fft.png')
    print("Phase screen plot saved as 'plots/physics/basic/phase_screen_fft.png'")


def test_non_kolmogorov(simulator: ChannelSimulator) -> None:
    """Test non-Kolmogorov turbulence spectra."""
    print("\n=== Non-Kolmogorov Turbulence Test ===")
    
    # Create frequency grid for PSD calculation
    N = simulator.phase_screen_resolution
    L = simulator.phase_screen_size
    df = 1.0 / L
    fx = np.arange(-N/2, N/2) * df
    fy = fx.copy()
    fx, fy = np.meshgrid(fx, fy)
    f = np.sqrt(fx**2 + fy**2)
    f[N//2, N//2] = 1e-10  # Avoid division by zero
    
    # Test different spectral indices
    spectral_indices = [3.0, 11/3, 4.0]
    labels = ['β=3.0 (Shallow)', 'β=11/3 (Kolmogorov)', 'β=4.0 (Steep)']
    colors = ['red', 'blue', 'green']
    
    plt.figure(figsize=(12, 8))
    
    for i, (beta, label, color) in enumerate(zip(spectral_indices, labels, colors)):
        # Set spectral index
        original_index = simulator.spectral_index
        simulator.spectral_index = beta
        
        # Generate PSD
        if beta == 11/3:
            psd = simulator._kolmogorov_psd(f, 0.1)
        else:
            psd = simulator._non_kolmogorov_psd(f, 0.1)
        
        # Take radial average for plotting
        radial_profile = []
        radial_freqs = []
        
        for radius in range(1, min(N//2, 100)):
            mask = (np.sqrt((fx)**2 + (fy)**2) >= radius-0.5) & \
                   (np.sqrt((fx)**2 + (fy)**2) < radius+0.5)
            if np.any(mask):
                radial_profile.append(np.mean(psd[mask]))
                radial_freqs.append(radius * df)
        
        # Plot
        plt.loglog(radial_freqs, radial_profile, color=color, linewidth=2, 
                  label=label, alpha=0.8)
        
        # Restore original spectral index
        simulator.spectral_index = original_index
    
    plt.xlabel('Spatial Frequency (cycles/m)')
    plt.ylabel('Power Spectral Density')
    plt.title('Non-Kolmogorov Turbulence Spectra Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('plots/physics/basic', exist_ok=True)
    plt.savefig('plots/physics/basic/non_kolmogorov_psd.png')
    print("Non-Kolmogorov PSD plot saved as 'plots/physics/basic/non_kolmogorov_psd.png'")


def test_multi_layer(simulator: ChannelSimulator) -> None:
    """Test multi-layer atmospheric modeling."""
    print("\n=== Multi-Layer Atmospheric Test ===")
    
    # Generate Cn2 profile
    distances = np.linspace(0, 10000, 100)[1:]  # Exclude zero
    cn2_profile = simulator._calculate_cn2_profile(distances)
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Cn2 profile
    plt.subplot(1, 3, 1)
    plt.semilogy(distances/1000, cn2_profile, 'b-', linewidth=2)
    plt.xlabel('Distance (km)')
    plt.ylabel('Cn² (m⁻²/³)')
    plt.title('Hufnagel-Valley Cn² Profile')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('plots/physics/basic', exist_ok=True)
    plt.savefig('plots/physics/basic/cn2_profile.png')
    print("Cn2 profile plot saved as 'plots/physics/basic/cn2_profile.png'")
    
    # Test multi-layer phase screens
    r0 = 0.1
    distance = 5000.0
    layer_counts = [1, 3, 5]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    original_layers = simulator.turbulence_layers
    
    for i, layers in enumerate(layer_counts):
        simulator.turbulence_layers = layers
        
        if layers == 1:
            phase_screen = simulator._generate_fft_phase_screen(r0, distance)
        else:
            phase_screen = simulator._generate_multi_layer_phase_screen(r0, distance)
        
        im = axes[i].imshow(phase_screen, cmap='RdBu_r', aspect='equal')
        axes[i].set_title(f'{layers} Layer{"s" if layers > 1 else ""}\nσ={np.std(phase_screen):.3f} rad')
        axes[i].set_xlabel('Pixels')
        axes[i].set_ylabel('Pixels')
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    
    simulator.turbulence_layers = original_layers
    
    plt.tight_layout()
    os.makedirs('plots/physics/basic', exist_ok=True)
    plt.savefig('plots/physics/basic/multi_layer_phase_screens.png')
    print("Multi-layer phase screen plot saved as 'plots/physics/basic/multi_layer_phase_screens.png'")


def test_aperture_averaging(simulator: ChannelSimulator) -> None:
    """Test aperture averaging effects."""
    print("\n=== Aperture Averaging Test ===")
    
    # Test aperture averaging factor vs Fried parameter
    r0_values = np.logspace(-2, 0, 50)
    aperture_diameters = [0.1, 0.3, 0.5, 1.0]
    
    plt.figure(figsize=(12, 8))
    
    original_diameter = simulator.receiver_aperture_diameter
    
    for diameter in aperture_diameters:
        simulator.receiver_aperture_diameter = diameter
        factors = []
        
        for r0 in r0_values:
            simulator.r0_current = r0
            simulator.last_scintillation_index = 0.5  # Moderate turbulence
            factor = simulator._calculate_enhanced_aperture_averaging_factor()
            factors.append(factor)
        
        plt.semilogx(r0_values, factors, linewidth=2, label=f'D = {diameter} m')
    
    simulator.receiver_aperture_diameter = original_diameter
    
    plt.xlabel('Fried Parameter r₀ (m)')
    plt.ylabel('Aperture Averaging Factor')
    plt.title('Aperture Averaging vs Turbulence Strength')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.tight_layout()
    os.makedirs('plots/physics/basic', exist_ok=True)
    plt.savefig('plots/physics/basic/aperture_averaging.png')
    print("Aperture averaging plot saved as 'plots/physics/basic/aperture_averaging.png'")


def test_inner_outer_scale(simulator: ChannelSimulator) -> None:
    """Test inner and outer scale effects."""
    print("\n=== Inner/Outer Scale Effects Test ===")
    
    # Create frequency grid
    N = simulator.phase_screen_resolution
    L = simulator.phase_screen_size
    df = 1.0 / L
    fx = np.arange(-N/2, N/2) * df
    fy = fx.copy()
    fx, fy = np.meshgrid(fx, fy)
    f = np.sqrt(fx**2 + fy**2)
    f[N//2, N//2] = 1e-10
    
    # Generate base Kolmogorov PSD
    base_psd = simulator._kolmogorov_psd(f, 0.1)
    
    # Save original values
    original_inner = simulator.inner_scale
    original_outer = simulator.outer_scale
    
    plt.figure(figsize=(15, 10))
    
    # Test different configurations
    configs = [
        {'inner': 0.0, 'outer': float('inf'), 'label': 'Pure Kolmogorov', 'style': 'k-'},
        {'inner': 0.001, 'outer': float('inf'), 'label': 'With Inner Scale (1mm)', 'style': 'r-'},
        {'inner': 0.0, 'outer': 100.0, 'label': 'With Outer Scale (100m)', 'style': 'b-'},
        {'inner': 0.001, 'outer': 100.0, 'label': 'von Karman (both scales)', 'style': 'g-'}
    ]
    
    for config in configs:
        simulator.inner_scale = config['inner']
        simulator.outer_scale = config['outer']
        
        # Apply scale modifications
        modified_psd = simulator._apply_scale_limits(f, base_psd.copy())
        
        # Calculate radial average
        radial_profile = []
        radial_freqs = []
        
        for radius in range(1, min(N//2, 100)):
            mask = (np.sqrt((fx)**2 + (fy)**2) >= radius-0.5) & \
                   (np.sqrt((fx)**2 + (fy)**2) < radius+0.5)
            if np.any(mask):
                radial_profile.append(np.mean(modified_psd[mask]))
                radial_freqs.append(radius * df)
        
        # Plot
        plt.loglog(radial_freqs, radial_profile, config['style'], 
                  linewidth=2, label=config['label'], alpha=0.8)
    
    # Restore original values
    simulator.inner_scale = original_inner
    simulator.outer_scale = original_outer
    
    plt.xlabel('Spatial Frequency (cycles/m)')
    plt.ylabel('Power Spectral Density')
    plt.title('Inner/Outer Scale Effects on Turbulence Spectrum')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('plots/physics/basic', exist_ok=True)
    plt.savefig('plots/physics/basic/inner_outer_scale.png')
    print("Inner/outer scale plot saved as 'plots/physics/basic/inner_outer_scale.png'")


def test_full_channel(simulator: ChannelSimulator) -> None:
    """Test the full channel simulation with advanced physics."""
    print("\n=== Full Channel Simulation Test ===")
    
    # Test SINR vs distance for different modes
    distances = np.linspace(50, 500, 50)
    modes = range(simulator.min_mode, simulator.max_mode + 1)
    
    plt.figure(figsize=(12, 8))
    
    for mode in modes:
        sinr_values = []
        
        for distance in distances:
            pos = np.array([distance, 0, 0])
            _, sinr_db = simulator.run_step(pos, mode)
            sinr_values.append(sinr_db)
        
        plt.plot(distances, sinr_values, linewidth=2, label=f'Mode {mode}', alpha=0.8)
    
    plt.xlabel('Distance (m)')
    plt.ylabel('SINR (dB)')
    plt.title('SINR vs Distance (Advanced Physics)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('plots/physics/basic', exist_ok=True)
    plt.savefig('plots/physics/basic/advanced_physics_sinr.png')
    print("Advanced physics SINR plot saved as 'plots/physics/basic/advanced_physics_sinr.png'")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test advanced physics implementation')
    parser.add_argument('--config', type=str, default='config/simulation_params.yaml',
                        help='Path to simulation configuration file')
    parser.add_argument('--test', type=str, default='all',
                        help='Test to run (fft, kolmogorov, layers, aperture, scales, channel, all)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create simulator
    simulator = ChannelSimulator(config)
    
    # Run tests
    if args.test in ['fft', 'all']:
        test_fft_phase_screen(simulator)
    
    if args.test in ['kolmogorov', 'all']:
        test_non_kolmogorov(simulator)
    
    if args.test in ['layers', 'all']:
        test_multi_layer(simulator)
    
    if args.test in ['aperture', 'all']:
        test_aperture_averaging(simulator)
    
    if args.test in ['scales', 'all']:
        test_inner_outer_scale(simulator)
    
    if args.test in ['channel', 'all']:
        test_full_channel(simulator)
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main() 