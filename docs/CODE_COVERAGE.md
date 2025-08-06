# Code Coverage Guide for OAM 6G

This document provides an overview of the code coverage system for the OAM 6G project, including how to run coverage analysis, interpret the results, and improve coverage.

## What is Code Coverage?

Code coverage is a measure of how much of your code is executed during your tests. It helps identify areas of your codebase that are not being tested, which might contain bugs or other issues.

## Coverage Tools

The OAM 6G project uses the following coverage tools:

- **coverage.py**: A Python tool for measuring code coverage
- **pytest-cov**: A pytest plugin for measuring coverage during test runs

## Running Coverage Analysis

### Basic Coverage Analysis

To run coverage analysis with pytest, use the following command:

```bash
./run_tests_with_coverage.sh
```

This will run all tests and generate an HTML coverage report in the `htmlcov/` directory. The report will be automatically opened in your default web browser.

### Running Coverage for Specific Tests

To run coverage analysis for specific test files or directories, use the following command:

```bash
python -m pytest tests/unit/ --cov=. --cov-report=html
```

### Coverage Report Formats

The coverage tool can generate reports in several formats:

- **HTML**: `--cov-report=html` (default in our scripts)
- **Terminal**: `--cov-report=term`
- **XML**: `--cov-report=xml`
- **JSON**: `--cov-report=json`

To generate multiple formats at once:

```bash
python -m pytest tests/ --cov=. --cov-report=html --cov-report=term --cov-report=xml
```

## Interpreting Coverage Reports

### HTML Report

The HTML report provides a detailed view of coverage for each file, highlighting which lines were executed (green) and which were not (red).

Key metrics in the report:

- **Statements**: The number of executable statements in the file
- **Missing**: The number of statements that were not executed during tests
- **Excluded**: The number of statements excluded from coverage analysis
- **Coverage**: The percentage of statements that were executed

### Terminal Report

The terminal report provides a summary of coverage for each file:

```
Name                         Stmts   Miss  Cover
------------------------------------------------
environment/__init__.py          5      0   100%
environment/oam_env.py         203     45    78%
models/__init__.py               4      0   100%
models/agent.py                 87     12    86%
...
------------------------------------------------
TOTAL                         1234    234    81%
```

## Coverage Configuration

Coverage settings are configured in two files:

- **.coveragerc**: Main configuration file for coverage.py
- **setup.cfg**: Additional configuration for pytest-cov

### Key Configuration Options

- **source**: Directories to include in coverage analysis
- **omit**: Files and directories to exclude from coverage analysis
- **exclude_lines**: Lines to exclude from coverage analysis (e.g., `if __name__ == "__main__":`)

## Improving Coverage

### Strategies for Improving Coverage

1. **Identify Uncovered Areas**: Use the coverage report to identify areas with low coverage
2. **Write Targeted Tests**: Write tests specifically for uncovered code
3. **Test Edge Cases**: Ensure your tests cover edge cases and error conditions
4. **Use Parametrized Tests**: Use pytest's parametrize feature to test multiple inputs
5. **Mock External Dependencies**: Use mocking to test code with external dependencies

### Example: Improving Coverage for a Function

If you have a function with low coverage:

```python
def calculate_throughput(sinr, bandwidth):
    """Calculate throughput based on SINR and bandwidth."""
    if sinr <= 0:
        return 0.0
    
    try:
        throughput = bandwidth * math.log2(1 + sinr)
        return throughput
    except Exception as e:
        logger.error(f"Error calculating throughput: {e}")
        return 0.0
```

Write tests that cover all branches:

```python
def test_calculate_throughput_positive_sinr():
    """Test throughput calculation with positive SINR."""
    throughput = calculate_throughput(10.0, 100e6)
    assert throughput > 0

def test_calculate_throughput_zero_sinr():
    """Test throughput calculation with zero SINR."""
    throughput = calculate_throughput(0.0, 100e6)
    assert throughput == 0.0

def test_calculate_throughput_negative_sinr():
    """Test throughput calculation with negative SINR."""
    throughput = calculate_throughput(-10.0, 100e6)
    assert throughput == 0.0

def test_calculate_throughput_error():
    """Test throughput calculation with error."""
    with patch('math.log2', side_effect=ValueError):
        throughput = calculate_throughput(10.0, 100e6)
        assert throughput == 0.0
```

## Coverage Targets

For IEEE journal publication readiness, the OAM 6G project aims for the following coverage targets:

- **Overall Coverage**: At least 80%
- **Core Components**: At least 90%
  - Channel Simulator
  - OAM Environment
  - Physics Calculations
- **Critical Functions**: 100%
  - SINR Calculation
  - Throughput Calculation
  - Handover Logic

## Continuous Integration

Coverage analysis is run automatically on each pull request and push to the main branch using GitHub Actions. The configuration is defined in `.github/workflows/coverage.yml`.

## Troubleshooting

### Common Issues

1. **Missing Coverage Data**: If your coverage report shows no data, ensure that your tests are actually running and executing code.

2. **Low Coverage Despite Tests**: Check if your tests are actually executing the code paths you expect. Use print statements or debuggers to verify.

3. **Excluded Files Still Appearing**: Check your `.coveragerc` and `setup.cfg` files to ensure the exclusion patterns are correct.

### Debugging Coverage

To debug coverage issues, you can run coverage in debug mode:

```bash
python -m coverage run --debug=sys,config tests/unit/test_file.py
```

## References

- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [IEEE Standard for Software Test Documentation](https://standards.ieee.org/standard/829-2008.html)