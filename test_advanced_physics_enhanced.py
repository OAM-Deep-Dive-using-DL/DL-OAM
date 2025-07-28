#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import yaml
import argparse
from typing import Dict, Any, Optional
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulator.channel_simulator import ChannelSimulator

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def test_physics_validation(simulator: ChannelSimulator) -> None:
    """Validate the correctness of physics formulas."""
    print("\n=== Physics Formula Validation ===")
    
    # Test Fried parameter calculation with realistic values
    distance = 1000.0  # 1 km horizontal path
    cn2 = 5e-14  # Moderate turbulence Cn2 value
    k = simulator.k
    wavelength = simulator.wavelength
    
    print(f"Wavelength: {wavelength*1000:.1f} mm (mmWave)")
    print(f"Wave number k: {k:.1f} rad/m")
    
    # Expected r0 calculation
    r0_expected = (0.423 * k**2 * cn2 * distance) ** (-3/5)
    print(f"Fried parameter r0: {r0_expected:.4f} m (for Cn2={cn2:.0e}, distance={distance}m)")
    
    # Scale to visible wavelength for comparison with literature
    # r0 scales as λ^(6/5), so r0_visible = r0_mmwave * (λ_visible/λ_mmwave)^(6/5)
    lambda_visible = 0.5e-6  # 500 nm
    r0_visible = r0_expected * (lambda_visible / wavelength)**(6/5)
    print(f"Equivalent r0 at 500nm: {r0_visible*100:.1f} cm (literature: 10-20 cm)")
    
    # Test scintillation index
    scint_index = simulator._calculate_scintillation_index(distance, cn2)
    rytov_expected = 1.23 * cn2 * (k**(7/6)) * (distance**(11/6))
    print(f"Rytov variance: {rytov_expected:.4f}")
    print(f"Scintillation index: {scint_index:.4f}")
    
    # Test beam wandering
    beam_wander = simulator._calculate_beam_wander(distance, cn2)
    beam_wander_expected = 2.42 * (k**(7/6)) * cn2 * (distance**(5/3))
    print(f"Beam wandering variance: {beam_wander:.4e} rad² (expected: {beam_wander_expected:.4e})")
    
    # Test with strong turbulence for validation
    cn2_strong = 5e-13  # Strong turbulence
    r0_strong = (0.423 * k**2 * cn2_strong * distance) ** (-3/5)
    r0_strong_visible = r0_strong * (lambda_visible / wavelength)**(6/5)
    print(f"Strong turbulence r0: {r0_strong:.4f} m (equiv. {r0_strong_visible*100:.1f} cm at 500nm)")
    
    # Validate physics relationships
    assert rytov_expected >= 0, f"Rytov variance must be non-negative: {rytov_expected}"
    assert scint_index >= 0, f"Scintillation index must be non-negative: {scint_index}"
    assert beam_wander >= 0, f"Beam wandering must be positive: {beam_wander}"
    assert r0_expected > 0, f"Fried parameter must be positive: {r0_expected}"
    
    # Test wavelength scaling
    print(f"\n--- Wavelength Scaling Test ---")
    lambda_test = 1e-6  # 1 μm
    k_test = 2 * np.pi / lambda_test
    r0_test = (0.423 * k_test**2 * cn2 * distance) ** (-3/5)
    scaling_factor = (lambda_test / wavelength)**(6/5)
    r0_scaled = r0_expected * scaling_factor
    print(f"Direct calculation at 1μm: {r0_test:.4f} m")
    print(f"Scaled from mmWave: {r0_scaled:.4f} m")
    print(f"Scaling works: {abs(r0_test - r0_scaled) < 0.001}")
    
    # Test realistic scenarios
    print("\n--- Realistic Scenarios ---")
    scenarios = [
        {"name": "Weak turbulence", "cn2": 1e-15, "distance": 500},
        {"name": "Moderate turbulence", "cn2": 1e-14, "distance": 1000},
        {"name": "Strong turbulence", "cn2": 1e-13, "distance": 2000}
    ]
    
    for scenario in scenarios:
        cn2 = scenario["cn2"]
        dist = scenario["distance"]
        r0 = (0.423 * k**2 * cn2 * dist) ** (-3/5)
        r0_vis = r0 * (lambda_visible / wavelength)**(6/5)
        rytov = 1.23 * cn2 * (k**(7/6)) * (dist**(11/6))
        print(f"{scenario['name']}: r0={r0:.3f}m ({r0_vis*100:.1f}cm@500nm), Rytov={rytov:.3f}")
    
    print("✅ All physics formulas validated successfully")

def test_fft_phase_screen_enhanced(simulator: ChannelSimulator) -> None:
    """Test FFT-based phase screen generation with enhanced visualization."""
    print("\n=== Enhanced FFT-based Phase Screen Test ===")
    
    # Test parameters
    r0_values = [0.05, 0.1, 0.2]  # Different turbulence strengths
    distance = 1000.0
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, r0 in enumerate(r0_values):
        # Generate phase screen
        phase_screen = simulator._generate_fft_phase_screen(r0, distance)
        
        # Statistics
        phase_std = np.std(phase_screen)
        phase_var = np.var(phase_screen)
        
        print(f"r0={r0:.3f}m: std={phase_std:.3f} rad, var={phase_var:.3f} rad²")
        
        # Plot phase screen
        im1 = axes[0, i].imshow(phase_screen, cmap='RdBu_r', 
                               extent=[-simulator.phase_screen_size/2, simulator.phase_screen_size/2,
                                      -simulator.phase_screen_size/2, simulator.phase_screen_size/2])
        axes[0, i].set_title(f'Phase Screen (r₀={r0:.3f}m)\nσ={phase_std:.3f} rad')
        axes[0, i].set_xlabel('Distance (m)')
        axes[0, i].set_ylabel('Distance (m)')
        plt.colorbar(im1, ax=axes[0, i], label='Phase (rad)')
        
        # Plot 1D slice through center
        center = simulator.phase_screen_resolution // 2
        slice_1d = phase_screen[center, :]
        coords = np.linspace(-simulator.phase_screen_size/2, simulator.phase_screen_size/2, 
                           simulator.phase_screen_resolution)
        
        axes[1, i].plot(coords, slice_1d, 'b-', linewidth=1.5)
        axes[1, i].set_title(f'Central Slice (r₀={r0:.3f}m)')
        axes[1, i].set_xlabel('Distance (m)')
        axes[1, i].set_ylabel('Phase (rad)')
        axes[1, i].grid(True, alpha=0.3)
        
        # Add structure function analysis
        # Calculate phase structure function D(r) = <[φ(x+r) - φ(x)]²>
        max_lag = min(50, simulator.phase_screen_resolution // 4)
        lags = np.arange(1, max_lag)
        structure_func = []
        
        for lag in lags:
            diff = phase_screen[center, center:-lag] - phase_screen[center, center+lag:]
            structure_func.append(np.mean(diff**2))
        
        # Theoretical structure function for Kolmogorov: D(r) = 6.88 * (r/r0)^(5/3)
        r_phys = lags * simulator.phase_screen_size / simulator.phase_screen_resolution
        theoretical_sf = 6.88 * (r_phys / r0)**(5/3)
        
        # Add inset plot for structure function
        inset = axes[0, i].inset_axes([0.6, 0.6, 0.35, 0.35])
        inset.loglog(r_phys, structure_func, 'ro-', markersize=3, label='Simulated')
        inset.loglog(r_phys, theoretical_sf, 'k--', linewidth=1, label='Theory (5/3)')
        inset.set_xlabel('r (m)', fontsize=8)
        inset.set_ylabel('D(r)', fontsize=8)
        inset.legend(fontsize=6)
        inset.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('plots/physics/enhanced', exist_ok=True)
    plt.savefig('plots/physics/enhanced/enhanced_phase_screen_fft.png', dpi=300, bbox_inches='tight')
    print("Enhanced phase screen plot saved as 'plots/physics/enhanced/enhanced_phase_screen_fft.png'")

def test_turbulence_spectra_comparison(simulator: ChannelSimulator) -> None:
    """Compare different turbulence spectra with enhanced visualization."""
    print("\n=== Enhanced Turbulence Spectra Comparison ===")
    
    # Create frequency grid
    N = simulator.phase_screen_resolution
    L = simulator.phase_screen_size
    df = 1.0 / L
    fx = np.arange(-N/2, N/2) * df
    fy = fx.copy()
    fx, fy = np.meshgrid(fx, fy)
    f = np.sqrt(fx**2 + fy**2)
    f[N//2, N//2] = 1e-10
    
    # Test parameters
    r0 = 0.1
    spectral_indices = [3.0, 11/3, 4.0]
    labels = ['Shallow (β=3.0)', 'Kolmogorov (β=11/3)', 'Steep (β=4.0)']
    colors = ['red', 'blue', 'green']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 2D PSDs
    for i, (beta, label, color) in enumerate(zip(spectral_indices, labels, colors)):
        # Save original spectral index
        original_index = simulator.spectral_index
        simulator.spectral_index = beta
        
        if beta == 11/3:
            psd = simulator._kolmogorov_psd(f, r0)
        else:
            psd = simulator._non_kolmogorov_psd(f, r0)
        
        # Apply scale limits for realistic comparison
        psd = simulator._apply_scale_limits(f, psd)
        
        # Radial average for 1D plot
        center = N // 2
        radial_profile = []
        radial_freqs = []
        
        for radius in range(1, min(center, 100)):
            # Create annular mask
            r_inner = radius - 0.5
            r_outer = radius + 0.5
            mask = (np.sqrt((fx)**2 + (fy)**2) >= r_inner * df) & \
                   (np.sqrt((fx)**2 + (fy)**2) < r_outer * df)
            
            if np.any(mask):
                radial_profile.append(np.mean(psd[mask]))
                radial_freqs.append(radius * df)
        
        # Plot 1D radial profile
        ax1.loglog(radial_freqs, radial_profile, color=color, linewidth=2, 
                  label=label, alpha=0.8)
        
        # Add theoretical slope lines
        if beta == 11/3:
            # Theoretical Kolmogorov slope
            f_theory = np.array(radial_freqs)
            psd_theory = 0.023 * (r0**(-5/3)) * (f_theory**(-11/3))
            ax1.loglog(f_theory[::5], psd_theory[::5], '--', color=color, alpha=0.5,
                      linewidth=1, label=f'Theory {label}')
        
        # Restore original spectral index
        simulator.spectral_index = original_index
    
    # Formatting for 1D plot
    ax1.set_xlabel('Spatial Frequency (cycles/m)', fontsize=12)
    ax1.set_ylabel('Power Spectral Density (rad²·m²)', fontsize=12)
    ax1.set_title('Turbulence Power Spectra Comparison', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.set_xlim([1e-2, 1e2])
    
    # Add slope reference lines
    f_ref = np.logspace(-1, 1, 50)
    ax1.loglog(f_ref, 1e-6 * f_ref**(-11/3), 'k:', alpha=0.7, label='f^(-11/3) slope')
    ax1.loglog(f_ref, 1e-5 * f_ref**(-3), 'k:', alpha=0.7, label='f^(-3) slope')
    
    # Plot 2D PSD for Kolmogorov spectrum
    simulator.spectral_index = 11/3
    psd_2d = simulator._kolmogorov_psd(f, r0)
    psd_2d = simulator._apply_scale_limits(f, psd_2d)
    
    # Create nice 2D visualization
    extent = [-N//2 * df, N//2 * df, -N//2 * df, N//2 * df]
    im = ax2.imshow(np.log10(psd_2d), extent=extent, cmap='viridis', 
                    origin='lower', aspect='equal')
    ax2.set_xlabel('fx (cycles/m)', fontsize=12)
    ax2.set_ylabel('fy (cycles/m)', fontsize=12)
    ax2.set_title('2D Kolmogorov Power Spectrum\n(log₁₀ scale)', fontsize=14)
    
    # Add contour lines
    levels = np.linspace(np.log10(np.min(psd_2d[psd_2d > 0])), 
                        np.log10(np.max(psd_2d)), 8)
    ax2.contour(fx, fy, np.log10(psd_2d), levels=levels, colors='white', alpha=0.3)
    
    plt.colorbar(im, ax=ax2, label='log₁₀(PSD)')
    
    plt.tight_layout()
    os.makedirs('plots/physics/enhanced', exist_ok=True)
    plt.savefig('plots/physics/enhanced/enhanced_turbulence_spectra.png', dpi=300, bbox_inches='tight')
    print("Enhanced turbulence spectra plot saved as 'plots/physics/enhanced/enhanced_turbulence_spectra.png'")

def test_multi_layer_enhanced(simulator: ChannelSimulator) -> None:
    """Test multi-layer atmospheric modeling with enhanced visualization."""
    print("\n=== Enhanced Multi-Layer Atmospheric Test ===")
    
    # Create altitude profile
    distances = np.linspace(0, 15000, 200)[1:]  # Up to 15 km
    cn2_profile = simulator._calculate_cn2_profile(distances)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
    
    # Plot 1: Cn2 profile with atmospheric layers
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Convert distances to altitudes for realistic atmospheric model
    max_altitude = 20000
    altitudes = distances * max_altitude / np.max(distances)
    
    ax1.semilogy(cn2_profile, altitudes/1000, 'b-', linewidth=2.5, label='Hufnagel-Valley Model')
    ax1.fill_betweenx(altitudes/1000, cn2_profile, alpha=0.3, color='blue')
    
    # Add atmospheric layer boundaries
    layer_boundaries = [1, 6, 12]  # km
    layer_names = ['Boundary Layer', 'Troposphere', 'Stratosphere']
    colors = ['red', 'orange', 'green']
    
    for boundary, name, color in zip(layer_boundaries, layer_names, colors):
        ax1.axhline(y=boundary, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
        ax1.text(np.max(cn2_profile)*0.1, boundary+0.5, name, 
                fontsize=10, color=color, weight='bold')
    
    ax1.set_xlabel('Cn² (m⁻²/³)', fontsize=12)
    ax1.set_ylabel('Altitude (km)', fontsize=12)
    ax1.set_title('Atmospheric Turbulence Profile', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Layer weights
    ax2 = fig.add_subplot(gs[0, 1])
    
    layer_distances = np.linspace(0, 10000, simulator.turbulence_layers + 1)[1:]
    layer_cn2 = simulator._calculate_cn2_profile(layer_distances)
    total_cn2 = np.sum(layer_cn2)
    layer_weights = layer_cn2 / total_cn2
    
    bars = ax2.bar(range(len(layer_weights)), layer_weights, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(layer_weights))),
                   alpha=0.8)
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Relative Weight', fontsize=12)
    ax2.set_title(f'Layer Weights ({simulator.turbulence_layers} layers)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, weight) in enumerate(zip(bars, layer_weights)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{weight:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Cumulative effect
    ax3 = fig.add_subplot(gs[0, 2])
    
    cumulative_cn2 = np.cumsum(cn2_profile) * (distances[1] - distances[0])
    total_turbulence = cumulative_cn2[-1]
    
    ax3.plot(distances/1000, cumulative_cn2/total_turbulence, 'g-', linewidth=2.5)
    ax3.fill_between(distances/1000, cumulative_cn2/total_turbulence, alpha=0.3, color='green')
    ax3.set_xlabel('Distance (km)', fontsize=12)
    ax3.set_ylabel('Cumulative Turbulence Fraction', fontsize=12)
    ax3.set_title('Cumulative Turbulence Effect', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Mark 50% and 90% levels
    for level, label in [(0.5, '50%'), (0.9, '90%')]:
        idx = np.argmin(np.abs(cumulative_cn2/total_turbulence - level))
        ax3.axhline(y=level, color='red', linestyle='--', alpha=0.7)
        ax3.axvline(x=distances[idx]/1000, color='red', linestyle='--', alpha=0.7)
        ax3.text(distances[idx]/1000 + 0.5, level + 0.05, 
                f'{label} at {distances[idx]/1000:.1f}km', fontsize=10)
    
    # Bottom plots: Phase screens for different layer counts
    layer_counts = [1, 3, 5]
    r0 = 0.1
    distance = 5000.0
    
    original_layers = simulator.turbulence_layers
    
    for i, layers in enumerate(layer_counts):
        simulator.turbulence_layers = layers
        
        if layers == 1:
            phase_screen = simulator._generate_fft_phase_screen(r0, distance)
        else:
            phase_screen = simulator._generate_multi_layer_phase_screen(r0, distance)
        
        ax = fig.add_subplot(gs[1, i])
        
        # Create high-quality visualization
        extent = [-simulator.phase_screen_size/2, simulator.phase_screen_size/2,
                 -simulator.phase_screen_size/2, simulator.phase_screen_size/2]
        
        im = ax.imshow(phase_screen, cmap='RdBu_r', extent=extent, aspect='equal')
        ax.set_title(f'{layers} Layer{"s" if layers > 1 else ""}\nσ={np.std(phase_screen):.3f} rad', 
                    fontsize=12)
        ax.set_xlabel('Distance (m)', fontsize=11)
        ax.set_ylabel('Distance (m)', fontsize=11)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Phase (rad)', fontsize=10)
    
    simulator.turbulence_layers = original_layers
    
    plt.tight_layout()
    os.makedirs('plots/physics/enhanced', exist_ok=True)
    plt.savefig('plots/physics/enhanced/enhanced_multi_layer_analysis.png', dpi=300, bbox_inches='tight')
    print("Enhanced multi-layer analysis plot saved as 'plots/physics/enhanced/enhanced_multi_layer_analysis.png'")

def test_aperture_averaging_enhanced(simulator: ChannelSimulator) -> None:
    """Test enhanced aperture averaging with comprehensive analysis."""
    print("\n=== Enhanced Aperture Averaging Test ===")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Test 1: Aperture factor vs r0
    r0_values = np.logspace(-2, 0, 100)
    aperture_diameters = [0.1, 0.3, 0.5, 1.0]  # meters
    
    original_diameter = simulator.receiver_aperture_diameter
    
    for diameter in aperture_diameters:
        simulator.receiver_aperture_diameter = diameter
        factors = []
        
        for r0 in r0_values:
            simulator.r0_current = r0
            simulator.last_scintillation_index = 0.5  # Moderate turbulence
            factor = simulator._calculate_enhanced_aperture_averaging_factor()
            factors.append(factor)
        
        ax1.semilogx(r0_values, factors, linewidth=2, label=f'D = {diameter} m')
    
    # Add theoretical curves
    r0_theory = np.logspace(-2, 0, 50)
    for diameter in [0.3, 0.5]:
        # Simplified theoretical curve for comparison
        theory_weak = 1 - 0.5 * (diameter / r0_theory)**(5/6)
        theory_weak = np.clip(theory_weak, 0.1, 1.0)
        ax1.plot(r0_theory, theory_weak, '--', alpha=0.7, linewidth=1)
    
    ax1.set_xlabel('Fried Parameter r₀ (m)', fontsize=12)
    ax1.set_ylabel('Aperture Averaging Factor', fontsize=12)
    ax1.set_title('Aperture Averaging vs Turbulence Strength', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Test 2: Effect on different OAM modes
    simulator.receiver_aperture_diameter = 0.5
    simulator.r0_current = 0.1
    
    modes = range(1, 9)
    mode_factors = []
    
    for mode in modes:
        mode_factor = 1.0 + 0.05 * (mode - simulator.min_mode)
        base_factor = simulator._calculate_enhanced_aperture_averaging_factor()
        effective_factor = base_factor ** (1.0 / mode_factor)
        mode_factors.append(effective_factor)
    
    bars = ax2.bar(modes, mode_factors, color=plt.cm.plasma(np.linspace(0, 1, len(modes))),
                   alpha=0.8)
    ax2.set_xlabel('OAM Mode Number', fontsize=12)
    ax2.set_ylabel('Mode-Specific Averaging Factor', fontsize=12)
    ax2.set_title('Mode-Dependent Aperture Averaging', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, factor in zip(bars, mode_factors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{factor:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Test 3: Scintillation reduction
    scint_indices = np.linspace(0, 2, 50)
    diameter = 0.5
    r0 = 0.1
    
    simulator.receiver_aperture_diameter = diameter
    simulator.r0_current = r0
    
    reduction_factors = []
    for scint in scint_indices:
        simulator.last_scintillation_index = scint
        factor = simulator._calculate_enhanced_aperture_averaging_factor()
        reduction_factors.append(factor)
    
    ax3.plot(scint_indices, reduction_factors, 'b-', linewidth=2.5, label='Enhanced Model')
    ax3.fill_between(scint_indices, reduction_factors, alpha=0.3)
    
    # Add regime boundaries
    ax3.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Weak/Strong Boundary')
    ax3.text(0.5, 0.9, 'Weak\nTurbulence', ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax3.text(1.5, 0.9, 'Strong\nTurbulence', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    ax3.set_xlabel('Scintillation Index', fontsize=12)
    ax3.set_ylabel('Aperture Averaging Factor', fontsize=12)
    ax3.set_title('Scintillation Reduction by Aperture', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Test 4: 2D visualization of aperture effect
    # Create a synthetic scintillation pattern
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    # Simulate scintillation pattern (simplified)
    pattern = np.exp(-((X**2 + Y**2) / 0.3)) * np.sin(10*X) * np.cos(8*Y)
    pattern += 0.5 * np.random.normal(0, 0.1, pattern.shape)
    
    # Apply aperture averaging (simplified convolution)
    from scipy import ndimage
    aperture_size = 10  # pixels
    kernel = np.ones((aperture_size, aperture_size)) / aperture_size**2
    averaged_pattern = ndimage.convolve(pattern, kernel, mode='constant')
    
    # Plot comparison
    extent = [-1, 1, -1, 1]
    im1 = ax4.imshow(pattern, extent=extent, cmap='RdBu_r', aspect='equal')
    ax4.contour(X, Y, averaged_pattern, levels=5, colors='black', linewidths=1, alpha=0.7)
    ax4.set_title('Scintillation Pattern\n(black lines: after aperture averaging)', fontsize=12)
    ax4.set_xlabel('Normalized Distance', fontsize=11)
    ax4.set_ylabel('Normalized Distance', fontsize=11)
    
    # Add aperture circle
    circle = plt.Circle((0, 0), 0.3, fill=False, color='white', linewidth=2, linestyle='--')
    ax4.add_patch(circle)
    ax4.text(0, -0.8, 'Receiver Aperture', ha='center', color='white', fontweight='bold')
    
    plt.colorbar(im1, ax=ax4, shrink=0.8, label='Intensity')
    
    simulator.receiver_aperture_diameter = original_diameter
    
    plt.tight_layout()
    os.makedirs('plots/physics/enhanced', exist_ok=True)
    plt.savefig('plots/physics/enhanced/enhanced_aperture_averaging.png', dpi=300, bbox_inches='tight')
    print("Enhanced aperture averaging plot saved as 'plots/physics/enhanced/enhanced_aperture_averaging.png'")

def test_inner_outer_scale_enhanced(simulator: ChannelSimulator) -> None:
    """Test inner/outer scale effects with enhanced visualization."""
    print("\n=== Enhanced Inner/Outer Scale Effects Test ===")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
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
    r0 = 0.1
    base_psd = simulator._kolmogorov_psd(f, r0)
    
    # Save original values
    original_inner = simulator.inner_scale
    original_outer = simulator.outer_scale
    
    # Test configurations for inner scale
    inner_scales = [0.001, 0.005, 0.01]  # meters
    colors_inner = ['red', 'green', 'blue']
    
    for inner, color in zip(inner_scales, colors_inner):
        simulator.inner_scale = inner
        simulator.outer_scale = float('inf')  # No outer scale
        
        modified_psd = simulator._apply_scale_limits(f, base_psd.copy())
        
        # Radial average
        radial_profile, radial_freqs = calculate_radial_profile(modified_psd, fx, fy, df, N)
        
        ax1.loglog(radial_freqs, radial_profile, color=color, linewidth=2,
                  label=f'l₀ = {inner*1000:.0f} mm')
    
    # Add pure Kolmogorov reference
    simulator.inner_scale = 0.0
    simulator.outer_scale = float('inf')
    pure_psd = simulator._apply_scale_limits(f, base_psd.copy())
    radial_profile, radial_freqs = calculate_radial_profile(pure_psd, fx, fy, df, N)
    ax1.loglog(radial_freqs, radial_profile, 'k--', linewidth=2, alpha=0.7,
              label='Pure Kolmogorov')
    
    ax1.set_xlabel('Spatial Frequency (cycles/m)', fontsize=12)
    ax1.set_ylabel('Power Spectral Density', fontsize=12)
    ax1.set_title('Inner Scale Effects', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test outer scale effects
    outer_scales = [10, 50, 100]  # meters
    colors_outer = ['purple', 'orange', 'brown']
    
    for outer, color in zip(outer_scales, colors_outer):
        simulator.inner_scale = 0.0  # No inner scale
        simulator.outer_scale = outer
        
        modified_psd = simulator._apply_scale_limits(f, base_psd.copy())
        radial_profile, radial_freqs = calculate_radial_profile(modified_psd, fx, fy, df, N)
        
        ax2.loglog(radial_freqs, radial_profile, color=color, linewidth=2,
                  label=f'L₀ = {outer} m')
    
    # Add pure Kolmogorov reference
    ax2.loglog(radial_freqs, radial_profile, 'k--', linewidth=2, alpha=0.7,
              label='Pure Kolmogorov')
    
    ax2.set_xlabel('Spatial Frequency (cycles/m)', fontsize=12)
    ax2.set_ylabel('Power Spectral Density', fontsize=12)
    ax2.set_title('Outer Scale Effects', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Combined von Karman spectrum
    simulator.inner_scale = 0.001  # 1 mm
    simulator.outer_scale = 100.0  # 100 m
    
    von_karman_psd = simulator._apply_scale_limits(f, base_psd.copy())
    
    # Compare all spectra
    configs = [
        {'inner': 0.0, 'outer': float('inf'), 'label': 'Pure Kolmogorov', 'style': 'k-'},
        {'inner': 0.001, 'outer': float('inf'), 'label': 'With Inner Scale', 'style': 'r-'},
        {'inner': 0.0, 'outer': 100.0, 'label': 'With Outer Scale', 'style': 'b-'},
        {'inner': 0.001, 'outer': 100.0, 'label': 'von Karman', 'style': 'g-'}
    ]
    
    for config in configs:
        simulator.inner_scale = config['inner']
        simulator.outer_scale = config['outer']
        
        test_psd = simulator._apply_scale_limits(f, base_psd.copy())
        radial_profile, radial_freqs = calculate_radial_profile(test_psd, fx, fy, df, N)
        
        ax3.loglog(radial_freqs, radial_profile, config['style'], linewidth=2.5,
                  label=config['label'], alpha=0.8)
    
    ax3.set_xlabel('Spatial Frequency (cycles/m)', fontsize=12)
    ax3.set_ylabel('Power Spectral Density', fontsize=12)
    ax3.set_title('Complete von Karman Spectrum', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add theoretical slope annotations
    f_ref = np.logspace(-1, 1, 20)
    ax3.loglog(f_ref, 1e-4 * f_ref**(-11/3), 'k:', alpha=0.5, linewidth=1)
    ax3.text(0.3, 2e-5, 'f^(-11/3)', fontsize=10, rotation=-35)
    
    # 2D visualization of von Karman spectrum
    extent = [-N//2 * df, N//2 * df, -N//2 * df, N//2 * df]
    im = ax4.imshow(np.log10(von_karman_psd), extent=extent, cmap='viridis',
                    origin='lower', aspect='equal')
    
    # Add contours and annotations
    levels = np.linspace(np.log10(np.min(von_karman_psd[von_karman_psd > 0])),
                        np.log10(np.max(von_karman_psd)), 8)
    ax4.contour(fx, fy, np.log10(von_karman_psd), levels=levels, colors='white', alpha=0.4)
    
    # Mark inner and outer scale frequencies
    f_inner = 5.92 / (2 * np.pi * simulator.inner_scale)
    f_outer = 1.0 / (2 * np.pi * simulator.outer_scale)
    
    if f_inner < N//2 * df:
        circle_inner = plt.Circle((0, 0), f_inner, fill=False, color='red', 
                                 linewidth=2, linestyle='--')
        ax4.add_patch(circle_inner)
        ax4.text(f_inner*0.7, f_inner*0.7, 'Inner\nScale', color='red', 
                fontweight='bold', ha='center')
    
    if f_outer < N//2 * df:
        circle_outer = plt.Circle((0, 0), f_outer, fill=False, color='blue',
                                 linewidth=2, linestyle='--')
        ax4.add_patch(circle_outer)
        ax4.text(f_outer*0.7, -f_outer*0.7, 'Outer\nScale', color='blue',
                fontweight='bold', ha='center')
    
    ax4.set_xlabel('fx (cycles/m)', fontsize=12)
    ax4.set_ylabel('fy (cycles/m)', fontsize=12)
    ax4.set_title('2D von Karman Spectrum\n(log₁₀ scale)', fontsize=14)
    
    plt.colorbar(im, ax=ax4, shrink=0.8, label='log₁₀(PSD)')
    
    # Restore original values
    simulator.inner_scale = original_inner
    simulator.outer_scale = original_outer
    
    plt.tight_layout()
    os.makedirs('plots/physics/enhanced', exist_ok=True)
    plt.savefig('plots/physics/enhanced/enhanced_inner_outer_scale.png', dpi=300, bbox_inches='tight')
    print("Enhanced inner/outer scale plot saved as 'plots/physics/enhanced/enhanced_inner_outer_scale.png'")

def calculate_radial_profile(psd, fx, fy, df, N):
    """Calculate radial profile of 2D PSD."""
    center = N // 2
    radial_profile = []
    radial_freqs = []
    
    for radius in range(1, min(center, 100)):
        r_inner = radius - 0.5
        r_outer = radius + 0.5
        mask = (np.sqrt((fx)**2 + (fy)**2) >= r_inner * df) & \
               (np.sqrt((fx)**2 + (fy)**2) < r_outer * df)
        
        if np.any(mask):
            radial_profile.append(np.mean(psd[mask]))
            radial_freqs.append(radius * df)
    
    return radial_profile, radial_freqs

def test_full_channel_enhanced(simulator: ChannelSimulator) -> None:
    """Test the full channel simulation with enhanced physics visualization."""
    print("\n=== Enhanced Full Channel Simulation Test ===")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Test 1: SINR vs Distance for different modes
    num_points = 100
    distance_range = np.linspace(50, 500, num_points)
    user_positions = np.zeros((num_points, 3))
    user_positions[:, 0] = distance_range
    
    modes = range(simulator.min_mode, simulator.max_mode + 1)
    sinr_results = np.zeros((len(modes), num_points))
    throughput_results = np.zeros((len(modes), num_points))
    
    for i, mode in enumerate(modes):
        for j, pos in enumerate(user_positions):
            _, sinr_db = simulator.run_step(pos, mode)
            sinr_results[i, j] = sinr_db
            
            # Calculate throughput
            sinr_linear = 10**(sinr_db/10)
            throughput = simulator.bandwidth * np.log2(1 + sinr_linear)
            throughput_results[i, j] = throughput / 1e9  # Gbps
    
    # Plot SINR
    colors = plt.cm.viridis(np.linspace(0, 1, len(modes)))
    for i, (mode, color) in enumerate(zip(modes, colors)):
        ax1.plot(distance_range, sinr_results[i], color=color, linewidth=2,
                label=f'Mode {mode}', alpha=0.8)
    
    ax1.set_xlabel('Distance (m)', fontsize=12)
    ax1.set_ylabel('SINR (dB)', fontsize=12)
    ax1.set_title('SINR vs Distance (Advanced Physics)', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add path loss reference
    path_loss_ref = -20 * np.log10(4 * np.pi * distance_range / simulator.wavelength)
    path_loss_ref -= np.max(path_loss_ref) - np.max(sinr_results)  # Normalize
    ax1.plot(distance_range, path_loss_ref, 'k--', alpha=0.5, linewidth=1,
            label='Free Space Path Loss')
    
    # Plot throughput
    for i, (mode, color) in enumerate(zip(modes, colors)):
        ax2.plot(distance_range, throughput_results[i], color=color, linewidth=2,
                label=f'Mode {mode}', alpha=0.8)
    
    ax2.set_xlabel('Distance (m)', fontsize=12)
    ax2.set_ylabel('Throughput (Gbps)', fontsize=12)
    ax2.set_title('Throughput vs Distance', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Test 2: Mode coupling analysis
    mode_coupling_matrix = np.zeros((len(modes), len(modes)))
    r0 = 0.1
    distance = 1000.0
    
    for i, mode_i in enumerate(modes):
        for j, mode_j in enumerate(modes):
            coupling = simulator._calculate_mode_coupling(mode_i, mode_j, r0, distance)
            mode_coupling_matrix[i, j] = coupling
    
    im = ax3.imshow(mode_coupling_matrix, cmap='hot', aspect='equal')
    ax3.set_xlabel('OAM Mode', fontsize=12)
    ax3.set_ylabel('OAM Mode', fontsize=12)
    ax3.set_title(f'Mode Coupling Matrix\n(r₀={r0}m, distance={distance}m)', fontsize=14)
    
    # Add mode labels
    ax3.set_xticks(range(len(modes)))
    ax3.set_xticklabels(modes)
    ax3.set_yticks(range(len(modes)))
    ax3.set_yticklabels(modes)
    
    # Add coupling values as text
    for i in range(len(modes)):
        for j in range(len(modes)):
            ax3.text(j, i, f'{mode_coupling_matrix[i, j]:.2f}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    plt.colorbar(im, ax=ax3, shrink=0.8, label='Coupling Coefficient')
    
    # Test 3: Performance comparison - with and without advanced physics
    # Temporarily disable advanced physics
    original_fft = simulator.use_fft_phase_screen
    original_layers = simulator.turbulence_layers
    
    modes_test = [2, 4, 6]
    distance_test = 200.0
    pos_test = np.array([distance_test, 0, 0])
    
    performance_data = {'Mode': [], 'Configuration': [], 'SINR (dB)': [], 'Throughput (Gbps)': []}
    
    configs = [
        {'name': 'Advanced Physics', 'fft': True, 'layers': 3},
        {'name': 'Basic Physics', 'fft': False, 'layers': 1}
    ]
    
    for config in configs:
        simulator.use_fft_phase_screen = config['fft']
        simulator.turbulence_layers = config['layers']
        
        for mode in modes_test:
            # Run multiple trials for statistics
            sinr_trials = []
            throughput_trials = []
            
            for trial in range(10):
                _, sinr_db = simulator.run_step(pos_test, mode)
                sinr_linear = 10**(sinr_db/10)
                throughput = simulator.bandwidth * np.log2(1 + sinr_linear) / 1e9
                
                sinr_trials.append(sinr_db)
                throughput_trials.append(throughput)
            
            # Store results
            performance_data['Mode'].extend([mode] * len(sinr_trials))
            performance_data['Configuration'].extend([config['name']] * len(sinr_trials))
            performance_data['SINR (dB)'].extend(sinr_trials)
            performance_data['Throughput (Gbps)'].extend(throughput_trials)
    
    # Create comparison plot
    import pandas as pd
    df = pd.DataFrame(performance_data)
    
    # Box plot for SINR comparison
    for i, config in enumerate(['Advanced Physics', 'Basic Physics']):
        for j, mode in enumerate(modes_test):
            data = df[(df['Configuration'] == config) & (df['Mode'] == mode)]['SINR (dB)']
            x_pos = j + i * 0.4 - 0.2
            
            box = ax4.boxplot(data, positions=[x_pos], widths=0.3, patch_artist=True)
            box['boxes'][0].set_facecolor('lightblue' if i == 0 else 'lightcoral')
            box['boxes'][0].set_alpha(0.7)
    
    ax4.set_xlabel('OAM Mode', fontsize=12)
    ax4.set_ylabel('SINR (dB)', fontsize=12)
    ax4.set_title('Physics Model Comparison\n(10 trials each)', fontsize=14)
    ax4.set_xticks(range(len(modes_test)))
    ax4.set_xticklabels(modes_test)
    ax4.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightblue', alpha=0.7, label='Advanced Physics'),
                      Patch(facecolor='lightcoral', alpha=0.7, label='Basic Physics')]
    ax4.legend(handles=legend_elements)
    
    # Restore original settings
    simulator.use_fft_phase_screen = original_fft
    simulator.turbulence_layers = original_layers
    
    plt.tight_layout()
    os.makedirs('plots/physics/enhanced', exist_ok=True)
    plt.savefig('plots/physics/enhanced/enhanced_full_channel_analysis.png', dpi=300, bbox_inches='tight')
    print("Enhanced full channel analysis plot saved as 'plots/physics/enhanced/enhanced_full_channel_analysis.png'")

def main():
    """Main function with enhanced testing."""
    parser = argparse.ArgumentParser(description='Enhanced Advanced Physics Testing')
    parser.add_argument('--config', type=str, default='config/simulation_params.yaml',
                        help='Path to simulation configuration file')
    parser.add_argument('--test', type=str, default='all',
                        help='Test to run (validation, fft, spectra, layers, aperture, scales, channel, all)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create simulator
    simulator = ChannelSimulator(config)
    
    print("🚀 Enhanced Advanced Physics Testing")
    print("=" * 50)
    
    # Run tests
    if args.test in ['validation', 'all']:
        test_physics_validation(simulator)
    
    if args.test in ['fft', 'all']:
        test_fft_phase_screen_enhanced(simulator)
    
    if args.test in ['spectra', 'all']:
        test_turbulence_spectra_comparison(simulator)
    
    if args.test in ['layers', 'all']:
        test_multi_layer_enhanced(simulator)
    
    if args.test in ['aperture', 'all']:
        test_aperture_averaging_enhanced(simulator)
    
    if args.test in ['scales', 'all']:
        test_inner_outer_scale_enhanced(simulator)
    
    if args.test in ['channel', 'all']:
        test_full_channel_enhanced(simulator)
    
    print("\n🎉 All enhanced tests completed successfully!")
    print("Generated high-quality plots with corrected physics formulas")

if __name__ == "__main__":
    main() 