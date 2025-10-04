# Publishing AgentFrame to PyPI

This guide provides step-by-step instructions for publishing the AgentFrame package to PyPI, making it available for installation via `pip install agentframe`.

## Prerequisites

Before publishing, ensure you have:

1. **PyPI Account**: Create accounts on both [TestPyPI](https://test.pypi.org/) and [PyPI](https://pypi.org/)
2. **API Tokens**: Generate API tokens for both platforms
3. **Build Tools**: Install required packaging tools
4. **Verified Package**: All tests pass and documentation is complete

## Setup Instructions

### 1. Install Build Tools

```bash
# Install/upgrade build tools
python -m pip install --upgrade build twine setuptools wheel

# Verify installation
python -m build --version
python -m twine --version
```

### 2. Configure PyPI Credentials

Create a `.pypirc` file in your home directory:

```bash
# Linux/macOS
nano ~/.pypirc

# Windows
notepad %USERPROFILE%\.pypirc
```

Add your credentials:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-testpypi-token-here
```

### 3. Get API Tokens

#### For TestPyPI:
1. Go to [TestPyPI Account Settings](https://test.pypi.org/manage/account/)
2. Scroll to "API tokens" section
3. Click "Add API token"
4. Name: "AgentFrame TestPyPI"
5. Scope: "Entire account" (for first upload)
6. Copy the token (starts with `pypi-`)

#### For PyPI:
1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Follow same steps as TestPyPI
3. Name: "AgentFrame PyPI"

## Pre-Publication Checklist

### 1. Version Management

Update version in `pyproject.toml`:

```toml
[project]
name = "agentframe"
version = "0.1.0"  # Update this
```

### 2. Update Changelog

Update `CHANGELOG.md` with new features, fixes, and changes:

```markdown
# Changelog

## [0.1.0] - 2024-10-04

### Added
- Initial release of AgentFrame
- Core agent framework with planning and replanning
- Support for OpenAI, Gemini, and Claude models
- Tool integration system with @tool decorator
- Chat history management
- Intent parsing
- Comprehensive documentation and examples
```

### 3. Run Quality Checks

```bash
# Run all tests
pytest tests/ -v --cov=agentframe --cov-report=html

# Type checking
mypy src/agentframe

# Code formatting
black src/ tests/ examples/

# Linting
ruff check src/ tests/ examples/

# Verify no syntax errors
python -m py_compile src/agentframe/__init__.py
```

### 4. Test Installation

```bash
# Build the package
python -m build

# Test installation in clean environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate
pip install dist/agentframe-0.1.0-py3-none-any.whl

# Test import
python -c "import agentframe; print(agentframe.__version__)"

# Clean up
deactivate
rm -rf test_env
```

## Publishing Process

### Step 1: Build the Package

```bash
# Clean previous builds
rm -rf dist/ build/ src/agentframe.egg-info/

# Build source distribution and wheel
python -m build

# Verify build artifacts
ls dist/
# Should see:
# agentframe-0.1.0.tar.gz
# agentframe-0.1.0-py3-none-any.whl
```

### Step 2: Upload to TestPyPI

```bash
# Upload to TestPyPI first
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --no-deps agentframe

# Test with dependencies
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple agentframe

# Verify it works
python -c "from agentframe import Agent; print('Import successful!')"
```

### Step 3: Upload to PyPI

```bash
# Upload to production PyPI
python -m twine upload dist/*

# Verify upload
pip install agentframe

# Test installation
python -c "import agentframe; print(f'AgentFrame v{agentframe.__version__} installed successfully!')"
```

## Post-Publication Steps

### 1. Create GitHub Release

```bash
# Create and push a git tag
git tag v0.1.0
git push origin v0.1.0

# Create release on GitHub
# Go to: https://github.com/agentframe/agentframe/releases/new
# - Tag version: v0.1.0
# - Release title: AgentFrame v0.1.0
# - Description: Copy from CHANGELOG.md
# - Attach dist/ files (optional)
```

### 2. Update Documentation

```bash
# Update installation instructions
# Update version references
# Publish documentation updates
```

### 3. Announce Release

- Update README.md with new version
- Post announcement in community channels
- Update any demo repositories

## Troubleshooting

### Common Issues

#### 1. Upload Failures

```bash
# Error: File already exists
# Solution: Increment version number in pyproject.toml

# Error: Invalid credentials
# Solution: Check .pypirc file and API tokens

# Error: Package name conflicts
# Solution: Choose a different package name
```

#### 2. Installation Issues

```bash
# Error: No module named 'agentframe'
# Check: Package was uploaded successfully
# Check: Using correct package name

# Error: ImportError during installation
# Check: All dependencies are correctly specified in pyproject.toml
# Check: Package structure is correct
```

#### 3. Version Conflicts

```bash
# Error: Version already exists
# Solution: Update version in pyproject.toml and rebuild

# Update version
sed -i 's/version = "0.1.0"/version = "0.1.1"/' pyproject.toml

# Rebuild and republish
rm -rf dist/
python -m build
python -m twine upload dist/*
```

## Automated Publishing

### GitHub Actions Workflow

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine

    - name: Build package
      run: python -m build

    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

Add `PYPI_API_TOKEN` to GitHub repository secrets.

## Version Management Strategy

### Semantic Versioning

Follow [SemVer](https://semver.org/):

- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.1.0): New features, backward compatible
- **PATCH** (0.0.1): Bug fixes, backward compatible

### Pre-release Versions

For beta releases:

```toml
version = "0.1.0b1"  # Beta 1
version = "0.1.0rc1" # Release candidate 1
```

### Development Versions

For development builds:

```toml
version = "0.1.0.dev1"  # Development build
```

## Package Maintenance

### Regular Updates

1. **Security Updates**: Monitor dependencies for vulnerabilities
2. **Dependency Updates**: Keep dependencies current
3. **Bug Fixes**: Address reported issues promptly
4. **Documentation**: Keep docs up to date

### Monitoring

- Watch PyPI download statistics
- Monitor GitHub issues and discussions
- Track user feedback and feature requests
- Monitor dependency security advisories

## Best Practices

1. **Test Thoroughly**: Always test on TestPyPI first
2. **Version Carefully**: Follow semantic versioning
3. **Document Changes**: Maintain detailed changelog
4. **Automate**: Use CI/CD for consistent releases
5. **Communicate**: Announce breaking changes clearly
6. **Support**: Respond to user issues promptly

## Resources

- [PyPI Documentation](https://packaging.python.org/)
- [Python Packaging Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)
- [Python Packaging Authority](https://www.pypa.io/)

## Getting Help

If you encounter issues during publishing:

1. Check the [Python Packaging Guide](https://packaging.python.org/)
2. Search [PyPI Help](https://pypi.org/help/)
3. Ask on [Python Packaging Discourse](https://discuss.python.org/c/packaging/)
4. Open an issue in the AgentFrame repository