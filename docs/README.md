# OAM 6G Handover with Deep Q-Learning

[![Tests](https://github.com/yourusername/oam-6g/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/oam-6g/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/yourusername/oam-6g/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/oam-6g)
[![Lint](https://github.com/yourusername/oam-6g/actions/workflows/lint.yml/badge.svg)](https://github.com/yourusername/oam-6g/actions/workflows/lint.yml)
[![Docs](https://github.com/yourusername/oam-6g/actions/workflows/docs.yml/badge.svg)](https://github.com/yourusername/oam-6g/actions/workflows/docs.yml)
[![Physics](https://github.com/yourusername/oam-6g/actions/workflows/physics.yml/badge.svg)](https://github.com/yourusername/oam-6g/actions/workflows/physics.yml)
[![Benchmarks](https://github.com/yourusername/oam-6g/actions/workflows/benchmark.yml/badge.svg)](https://github.com/yourusername/oam-6g/actions/workflows/benchmark.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
pip install -r config/requirements.txt
pip install pytest pytest-cov coverage
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

### Running Tests
```bash
# Run all tests
./run_tests.sh

# Run specific test categories
./run_tests.sh -t unit
./run_tests.sh -t integration
./run_tests.sh -t physics
./run_tests.sh -t regression

# Run tests with verbose output
./run_tests.sh -v
```

### Running Tests with Coverage
```bash
# Run all tests with coverage
./run_tests_with_coverage.sh

# Run specific tests with coverage
python -m pytest tests/unit/ --cov=. --cov-report=html
```

## Project Structure

```
├── simulator/                 # Channel simulation modules
│   └── channel_simulator.py   # Main physics-based simulator
├── environment/               # RL environment
│   ├── oam_env.py             # Base Gymnasium environment wrapper
│   └── stable_oam_env.py      # Stable reward environment
├── models/                    # Neural network models
│   ├── dqn_model.py           # DQN architecture
│   └── agent.py               # RL agent implementation
├── config/                    # Configuration files
│   ├── simulation_params.yaml # Simulation parameters
│   ├── base_config_new.yaml   # Base configuration
│   ├── rl_config_new.yaml     # RL configuration
│   └── extended_training_config.yaml # Extended training configuration
├── utils/                     # Utility functions
│   ├── visualization.py       # Plotting and visualization
│   ├── config_utils.py        # Configuration utilities
│   └── hierarchical_config.py # Hierarchical configuration system
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   ├── physics/               # Physics tests
│   └── regression/            # Regression tests
├── plots/                     # 📊 All generated visualizations
│   ├── enhanced_*.png         # Enhanced physics plots
│   ├── physics/               # Physics validation plots
│   ├── training/              # Training progress plots
│   ├── evaluation/            # Model evaluation plots
│   └── analysis/              # Performance analysis plots
├── results/                   # Training results and logs
├── docs/                      # Documentation
│   ├── README.md              # Main documentation
│   ├── CODE_COVERAGE.md       # Coverage documentation
│   └── TESTING.md             # Testing documentation
├── .github/workflows/         # CI/CD workflows
│   ├── tests.yml              # Test workflow
│   ├── coverage.yml           # Coverage workflow
│   ├── lint.yml               # Linting workflow
│   ├── physics.yml            # Physics validation workflow
│   ├── docs.yml               # Documentation workflow
│   ├── environment.yml        # Environment verification workflow
│   └── benchmark.yml          # Performance benchmarking workflow
├── main.py                    # Main entry point
├── run_tests.sh               # Test runner script
├── run_tests_with_coverage.sh # Coverage runner script
└── .coveragerc                # Coverage configuration
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
