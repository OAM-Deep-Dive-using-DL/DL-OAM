# Reproducibility Guide for OAM 6G Research

## Overview

This document outlines the measures taken to ensure reproducibility of the OAM 6G research results, which is essential for IEEE journal publication standards.

## Version Pinning

### Why Version Pinning Matters

In scientific computing and machine learning research, reproducibility is a critical requirement. Different versions of libraries can produce different results due to:

1. **Algorithm Changes**: Newer versions may implement different algorithms or optimizations
2. **Default Parameter Changes**: Default parameters might change between versions
3. **Bug Fixes**: Fixes in newer versions might change behavior
4. **API Changes**: Function signatures or return values might change
5. **Dependency Chain Effects**: A change in one library can cascade through dependencies

For IEEE journal publications, it's essential that other researchers can reproduce your exact results.

### Implementation

We've implemented strict version pinning in our `requirements.txt` file:

```
numpy==1.26.4
scipy==1.13.1
torch==2.7.1
torchvision==0.22.1
torchaudio==2.7.1
gymnasium==1.2.0
plotly==5.22.0
pyyaml==6.0.1
matplotlib==3.9.2
seaborn==0.13.2
tensorboard==2.19.0
tensorboard-data-server==0.7.2
```

This ensures that anyone setting up the environment will use exactly the same library versions that were used during development and evaluation.

## Random Seed Control

In addition to version pinning, we control randomness by:

1. Setting fixed random seeds for Python's `random`, NumPy, PyTorch, and environment seeds
2. Using deterministic algorithms where possible
3. Documenting any non-deterministic components

Example from our code:

```python
def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

## Hardware Considerations

Different hardware can affect results, especially with:

- Floating-point precision differences
- GPU vs. CPU implementations
- Parallelization effects

We document our development and evaluation hardware in the publication and note any hardware-specific considerations.

## Environment Variables

Some libraries use environment variables that can affect results. We document any relevant environment variables in the setup instructions.

## Installation Instructions

For reproducible setup, we recommend using a virtual environment:

```bash
# Create a virtual environment
python -m venv oam6g_env

# Activate the environment
source oam6g_env/bin/activate  # On Windows: oam6g_env\Scripts\activate

# Install exact package versions
pip install -r config/requirements.txt
```

## Verification

To verify your setup matches our development environment, run:

```bash
python scripts/verification/verify_environment.py
```

This script checks that all packages are installed with the correct versions and that the environment is properly configured.

## IEEE Journal Publication Standards

IEEE journals typically require:

1. Clear documentation of the experimental setup
2. Sufficient information to reproduce results
3. Availability of code and data (when possible)
4. Statistical analysis of results
5. Comparison with baseline methods

Our version pinning and reproducibility measures ensure compliance with these standards.

## References

1. Pineau, J., Vincent-Lamarre, P., Sinha, K. et al. "Improving Reproducibility in Machine Learning Research." arXiv:2003.12206, 2020.
2. Gundersen, O.E., Kjensmo, S. "State of the Art: Reproducibility in Artificial Intelligence." AAAI Conference on Artificial Intelligence, 2018.
3. IEEE Editorial Board. "IEEE Editorial Style Manual." IEEE, 2021.