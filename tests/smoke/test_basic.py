"""
Basic smoke test without external dependencies.

Tests core Python functionality and project structure.
"""

import os
from pathlib import Path


def test_project_structure():
    """Test that the basic project structure exists."""
    project_root = Path(__file__).parent.parent.parent
    
    # Check that essential directories exist
    assert (project_root / "src").exists()
    assert (project_root / "tests").exists()
    assert (project_root / "data").exists()
    assert (project_root / "models").exists()
    
    # Check that essential files exist
    assert (project_root / "README.md").exists()
    assert (project_root / "requirements.txt").exists()
    assert (project_root / "setup.py").exists()


def test_src_module_structure():
    """Test that the src module structure is correct."""
    project_root = Path(__file__).parent.parent.parent
    src_dir = project_root / "src"
    
    # Check that submodules exist
    assert (src_dir / "utils").exists()
    assert (src_dir / "data_collection").exists() 
    assert (src_dir / "embeddings").exists()
    assert (src_dir / "classification").exists()
    assert (src_dir / "post_processing").exists()
    
    # Check that __init__.py files exist
    assert (src_dir / "__init__.py").exists()
    assert (src_dir / "utils" / "__init__.py").exists()


def test_requirements_file():
    """Test that requirements.txt contains expected dependencies."""
    project_root = Path(__file__).parent.parent.parent
    req_file = project_root / "requirements.txt"
    
    content = req_file.read_text()
    
    # Check for key dependencies
    assert "scikit-learn" in content
    assert "pandas" in content
    assert "numpy" in content
    assert "pytest" in content


def test_python_syntax():
    """Test that key Python files have valid syntax."""
    project_root = Path(__file__).parent.parent.parent
    
    # Test that we can compile key Python files
    python_files = [
        "src/utils/error_types.py",
        "src/utils/config.py",
        "setup.py"
    ]
    
    for file_path in python_files:
        full_path = project_root / file_path
        if full_path.exists():
            with open(full_path, 'r') as f:
                content = f.read()
            # This will raise SyntaxError if invalid
            compile(content, str(full_path), 'exec')


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
