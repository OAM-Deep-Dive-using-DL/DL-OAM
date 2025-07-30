# OAM 6G Handover with Deep Q-Learning

This project implements a Deep Q-Network (DQN) for optimizing OAM (Orbital Angular Momentum) mode handover in 6G wireless networks. The system uses reinforcement learning to intelligently switch between OAM modes based on channel conditions, user mobility, and network performance.

## Features

- **Deep Q-Network (DQN)** for intelligent OAM mode selection
- **High-fidelity channel simulator** with realistic atmospheric effects
- **Advanced physics modeling** including:
  - FFT-based phase screen generation (McGlamery method)
  - Non-Kolmogorov turbulence support
  - Multi-layer atmospheric modeling
  - Enhanced aperture averaging
  - Inner/outer scale turbulence effects
- **Gymnasium-compatible environment** for RL training
- **Comprehensive evaluation** with performance metrics
- **Interactive visualizations** for analysis

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv oam_rl_env
source oam_rl_env/bin/activate  # On Windows: oam_rl_env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python main.py train
```

### Evaluation
```bash
python main.py evaluate
```

### Testing Advanced Physics
```bash
python test_advanced_physics_enhanced.py --test all
```

### Basic Physics Testing
```bash
python test_advanced_physics.py --all
```

## Project Structure

```
├── simulator/                 # Channel simulation modules
│   └── channel_simulator.py   # Main physics-based simulator
├── environment/               # RL environment
│   └── oam_env.py             # Gymnasium environment wrapper
├── models/                    # Neural network models
│   ├── dqn_model.py          # DQN architecture
│   └── agent.py              # RL agent implementation
├── config/                    # Configuration files
│   └── simulation_params.yaml # Simulation parameters
├── utils/                     # Utility functions
│   └── visualization.py      # Plotting and visualization
├── plots/                     # 📊 All generated visualizations
│   ├── enhanced_*.png         # Enhanced physics plots
│   ├── physics/              # Physics validation plots
│   ├── training/             # Training progress plots
│   ├── evaluation/           # Model evaluation plots
│   └── analysis/             # Performance analysis plots
├── results/                   # Training results and logs
├── main.py                   # Main entry point
├── train_rl.py              # Training script
├── evaluate_rl.py           # Evaluation script
└── organize_plots.py        # Plot organization utility
```

## Plots Directory

All visualizations are centrally managed in the `plots/` directory:

- **Enhanced Physics Plots**: High-quality visualizations with corrected physics formulas
- **Basic Physics Plots**: Standard validation plots
- **Training Plots**: RL training progress and metrics
- **Evaluation Plots**: Model performance analysis
- **Comparison Plots**: Comparative studies and ablations

Use `python organize_plots.py --list` to see all available plots.

## Advanced Physics Features

The simulator includes state-of-the-art atmospheric modeling:

1. **FFT-based Phase Screen Generation**: McGlamery method for accurate turbulence simulation
2. **Non-Kolmogorov Turbulence**: Configurable spectral indices beyond standard Kolmogorov
3. **Multi-layer Atmospheric Modeling**: Hufnagel-Valley profile with altitude-dependent effects
4. **Enhanced Aperture Averaging**: Andrews & Phillips model with inner/outer scale corrections
5. **Inner/Outer Scale Effects**: von Karman spectrum with finite turbulence scales

See `ADVANCED_PHYSICS_IMPLEMENTATION.md` for detailed technical documentation.

## Configuration

The system is highly configurable through YAML files:

- `config/simulation_params.yaml`: Main simulation parameters
- `config/advanced_physics.yaml`: Advanced physics settings

Key parameters include:
- OAM mode range and spacing
- Channel model parameters
- Turbulence characteristics
- Advanced physics features
- RL hyperparameters

## Results

The system generates comprehensive results including:

- Training progress and convergence metrics
- Performance comparisons across different OAM modes
- Channel quality analysis (SINR, throughput)
- Handover statistics and efficiency
- High-quality publication-ready visualizations

## Physics Validation

All physics formulas have been validated against literature:

- **Fried Parameter**: Correct λ^(6/5) wavelength scaling
- **Scintillation Index**: Rytov variance calculations
- **Mode Coupling**: Physics-based selection rules
- **Aperture Averaging**: Theoretical model compliance

Run `python test_advanced_physics_enhanced.py --test validation` for detailed validation.

## Contributing

1. Follow the existing code structure and documentation style
2. Add tests for new physics models
3. Update configuration files as needed
4. Generate appropriate visualizations
5. Update documentation

## References

1. Fried, D. L. (1966). "Optical Resolution Through a Randomly Inhomogeneous Medium"
2. Andrews, L. C., & Phillips, R. L. (2005). "Laser beam propagation through random media"
3. Lane, R. G., et al. (1992). "Simulation of a Kolmogorov phase screen"
4. Schmidt, J. D. (2010). "Numerical simulation of optical wave propagation"
5. Hardy, J. W. (1998). "Adaptive optics for astronomical telescopes" 
