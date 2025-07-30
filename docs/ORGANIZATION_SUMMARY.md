# File Organization Summary

The project files have been reorganized into a logical folder structure for better maintainability and clarity.

## New Folder Structure

### 📁 `scripts/` - All executable scripts
- **`main.py`** - Main entry point for the application
- **`training/`** - Training-related scripts
  - `train_rl.py` - Standard DQN training
  - `train_stable_rl.py` - Training with stable rewards
- **`evaluation/`** - Evaluation and testing scripts
  - `evaluate_rl.py` - Model evaluation and performance analysis
- **`analysis/`** - Data analysis and comparison scripts
  - `compare_rewards.py` - Compare training results and rewards
- **`visualization/`** - Plotting and visualization scripts
  - `visualize_oam_modes.py` - OAM mode visualization
- **`verification/`** - Physics and model verification scripts
  - `verify_oam_physics.py` - OAM physics validation

### 📁 `docs/` - Documentation
- **`README.md`** - Main project documentation

### 📁 `config/` - Configuration files
- **`requirements.txt`** - Python dependencies
- **`rl_params.yaml`** - Reinforcement learning parameters
- **`simulation_params.yaml`** - Simulation configuration
- **`stable_reward_params.yaml`** - Stable reward configuration

## Benefits of This Organization

1. **Logical Grouping**: Related files are grouped together
2. **Easy Navigation**: Clear separation of concerns
3. **Maintainability**: Easier to find and modify specific functionality
4. **Scalability**: Easy to add new scripts to appropriate folders
5. **Documentation**: Clear structure for new contributors

## Usage Examples

```bash
# Training
python scripts/training/train_rl.py
python scripts/training/train_stable_rl.py

# Evaluation
python scripts/evaluation/evaluate_rl.py --model-dir results/train_xxx

# Analysis
python scripts/analysis/compare_rewards.py

# Visualization
python scripts/visualization/visualize_oam_modes.py

# Verification
python scripts/verification/verify_oam_physics.py

# Main entry point
python scripts/main.py train
python scripts/main.py evaluate --model-dir results/train_xxx
```

## Original Files Moved

- `train_rl.py` → `scripts/training/train_rl.py`
- `train_stable_rl.py` → `scripts/training/train_stable_rl.py`
- `evaluate_rl.py` → `scripts/evaluation/evaluate_rl.py`
- `compare_rewards.py` → `scripts/analysis/compare_rewards.py`
- `visualize_oam_modes.py` → `scripts/visualization/visualize_oam_modes.py`
- `verify_oam_physics.py` → `scripts/verification/verify_oam_physics.py`
- `README.md` → `docs/README.md`
- `main.py` → `scripts/main.py`
- `requirements.txt` → `config/requirements.txt`

All file contents remain unchanged - only the locations have been reorganized for better structure. 