# Continuous Integration and Continuous Deployment (CI/CD)

This document describes the CI/CD setup for the OAM 6G project, which is designed to ensure code quality, scientific accuracy, and reproducibility for IEEE journal publication standards.

## GitHub Actions Workflows

The project uses GitHub Actions for CI/CD with the following workflows:

### 1. Tests Workflow (`tests.yml`)

Runs all tests to verify code functionality:

- **Trigger**: On push to main branch or any pull request to main
- **Python Versions**: 3.8, 3.9, 3.10
- **Steps**:
  - Run unit tests
  - Run integration tests
  - Run regression tests
  - Generate test report
  - Upload test results as artifacts

### 2. Coverage Workflow (`coverage.yml`)

Generates code coverage reports:

- **Trigger**: On push to main branch or any pull request to main
- **Python Version**: 3.10
- **Steps**:
  - Generate coverage reports for unit tests
  - Generate coverage reports for integration tests
  - Generate coverage reports for physics tests
  - Combine coverage reports
  - Upload to Codecov
  - Upload HTML report as artifact

### 3. Lint Workflow (`lint.yml`)

Checks code quality and style:

- **Trigger**: On push to main branch or any pull request to main
- **Python Version**: 3.10
- **Steps**:
  - Check with flake8 for syntax errors
  - Check formatting with black
  - Check imports with isort
  - Type check with mypy

### 4. Physics Validation Workflow (`physics.yml`)

Validates physics calculations against literature:

- **Trigger**: On push to main or PR that changes simulator, physics calculator, or physics tests
- **Python Version**: 3.10
- **Steps**:
  - Run physics tests
  - Validate against literature
  - Generate physics validation report

### 5. Documentation Workflow (`docs.yml`)

Builds and deploys documentation:

- **Trigger**: On push to main or PR that changes docs or Python files
- **Python Version**: 3.10
- **Steps**:
  - Build documentation with Sphinx
  - Upload documentation artifact
  - Deploy to GitHub Pages (main branch only)

### 6. Environment Verification Workflow (`environment.yml`)

Verifies environment setup and package installation:

- **Trigger**: On push to main or PR that changes requirements or package files
- **Python Versions**: 3.8, 3.9, 3.10
- **Steps**:
  - Verify environment with verify_environment.py
  - Test installation as package

### 7. Performance Benchmarks Workflow (`benchmark.yml`)

Runs performance benchmarks:

- **Trigger**: On push to main or PR that changes simulator, environment, or models
- **Python Version**: 3.10
- **Steps**:
  - Run simulator benchmarks
  - Run environment benchmarks
  - Run agent benchmarks
  - Compare with previous benchmarks
  - Alert on performance regression

## Setting Up Local Development

To ensure your code will pass CI checks before pushing:

1. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Run linting locally:
   ```bash
   flake8 .
   black --check .
   isort --check .
   mypy --ignore-missing-imports environment models simulator utils
   ```

3. Run tests locally:
   ```bash
   pytest tests/unit/
   pytest tests/integration/
   pytest tests/physics/
   pytest tests/regression/
   ```

4. Check coverage locally:
   ```bash
   pytest --cov=. --cov-report=html
   ```

5. Run benchmarks locally:
   ```bash
   pytest tests/benchmarks/ --benchmark-only
   ```

## CI/CD Pipeline Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Code Push  │────▶│ Lint Check  │────▶│    Tests    │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Deploy    │◀────│    Build    │◀────│   Coverage   │
│    Docs     │     │    Docs     │     │   Report    │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
                                        ┌─────────────┐
                                        │   Physics   │
                                        │ Validation  │
                                        └─────────────┘
                                              │
                                              ▼
                                        ┌─────────────┐
                                        │ Performance │
                                        │ Benchmarks  │
                                        └─────────────┘
```

## Continuous Deployment

When changes are merged to the main branch:

1. Documentation is automatically deployed to GitHub Pages
2. Performance benchmarks are tracked over time
3. Coverage reports are uploaded to Codecov

## Badges

The following badges are available for the README:

- ![Tests](https://github.com/yourusername/oam-6g/actions/workflows/tests.yml/badge.svg)
- ![Coverage](https://codecov.io/gh/yourusername/oam-6g/branch/main/graph/badge.svg)
- ![Lint](https://github.com/yourusername/oam-6g/actions/workflows/lint.yml/badge.svg)
- ![Docs](https://github.com/yourusername/oam-6g/actions/workflows/docs.yml/badge.svg)
- ![Physics](https://github.com/yourusername/oam-6g/actions/workflows/physics.yml/badge.svg)