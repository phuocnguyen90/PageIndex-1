"""
Pytest configuration and shared fixtures for PageIndex tests.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()


def pytest_configure(config):
    """Configure pytest markers and settings."""
    config.addinivalue_line(
        "markers", "api: marks tests as API calls (deselect with '-m \"not api\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# Shared fixtures
@pytest.fixture
def sample_pdf_path():
    """Path to a sample PDF file for testing."""
    pdf_dir = Path(__file__).parent / "pdfs"
    # Return first available PDF
    for pdf_file in pdf_dir.glob("*.pdf"):
        return str(pdf_file)
    return None


@pytest.fixture
def sample_md_path():
    """Path to a sample Markdown file for testing."""
    md_file = Path(__file__).parent / "sample.md"
    if md_file.exists():
        return str(md_file)
    return None


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for test results."""
    output_dir = tmp_path / "results"
    output_dir.mkdir(exist_ok=True)
    return str(output_dir)


@pytest.fixture
def reset_cost_tracker():
    """Reset the global cost tracker before/after tests."""
    from pageindex.cost_tracker import reset_global_tracker
    reset_global_tracker()
    yield
    reset_global_tracker()


# Import pytest for fixtures
import pytest
