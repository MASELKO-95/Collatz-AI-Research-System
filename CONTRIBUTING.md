# Contributing to Collatz AI Research System

Thank you for your interest in contributing to the Collatz AI Research System! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/collatz-ai.git
   cd collatz-ai
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/collatz-ai.git
   ```

## Development Setup

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.4+ (for GPU training)
- g++ compiler for C++ modules
- 16GB+ RAM recommended

### Installation

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8 mypy pre-commit
   ```

3. **Compile C++ modules**:
   ```bash
   cd src
   g++ -shared -fPIC -O3 -o libcollatz.so collatz_core.cpp
   g++ -shared -fPIC -O3 -std=c++11 -pthread -o libloop_searcher.so loop_searcher.cpp
   cd ..
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## How to Contribute

### Areas of Contribution

We welcome contributions in the following areas:

- **Model Architecture**: Improvements to the Transformer model
- **Data Generation**: Faster or more efficient data generation
- **Loop Detection**: Better algorithms for cycle detection
- **Documentation**: Tutorials, examples, API docs
- **Testing**: Additional test cases and coverage
- **Visualization**: Better analysis and visualization tools
- **Performance**: Optimization and profiling

### Workflow

1. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Write tests** for new functionality

4. **Run tests** to ensure everything works:
   ```bash
   pytest tests/ -v
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## Coding Standards

### Python Code Style

- **Formatting**: Use [Black](https://black.readthedocs.io/) with 100 character line length
- **Linting**: Code must pass [flake8](https://flake8.pycqa.org/) checks
- **Type Hints**: Add type annotations to all functions
- **Docstrings**: Use Google-style docstrings

Example:
```python
def get_stopping_time(n: int) -> int:
    """
    Calculate the stopping time for a number in the Collatz sequence.

    Args:
        n: Starting number (must be positive)

    Returns:
        Number of steps to reach 1

    Raises:
        ValueError: If n is not positive
    """
    if n <= 0:
        raise ValueError("n must be positive")
    # Implementation...
```

### C++ Code Style

- Use C++11 or later
- Follow standard C++ naming conventions
- Add comments for complex algorithms
- Optimize for performance where appropriate

### Running Code Quality Checks

Before committing, run:

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/ --ignore-missing-imports

# Run all pre-commit hooks
pre-commit run --all-files
```

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names
- Test both success and failure cases
- Aim for >80% code coverage

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_engine.py -v

# Run specific test
pytest tests/test_engine.py::TestGetStoppingTime::test_known_stopping_times -v
```

## Pull Request Process

1. **Update documentation** if you've changed APIs or added features
2. **Add tests** for new functionality
3. **Update CHANGELOG** (if applicable)
4. **Ensure CI passes** - all tests and checks must pass
5. **Request review** from maintainers
6. **Address feedback** promptly and professionally

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] CI/CD pipeline passes
- [ ] Commit messages are clear and descriptive

## Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md) when filing issues.

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU, etc.)
- Error messages and stack traces

## Suggesting Features

Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md).

Include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (if you have ideas)
- Potential drawbacks or alternatives

## Questions?

- **GitHub Discussions**: For general questions and discussions
- **GitHub Issues**: For bug reports and feature requests
- **Pull Requests**: For code contributions

## License

By contributing, you agree that your contributions will be licensed under the GNU General Public License v3.0.

---

Thank you for contributing to the Collatz AI Research System! 🎯
