import pytest
from src.utils.dataset_loader import read_csv_flexible

BLOB_URL = "https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification/blob/feat/initial-error-table/data/ground_truth_qna.csv"

def test_local_csv_loading():
    """Test loading a local CSV file."""
    df = read_csv_flexible("data/math20.csv")
    assert not df.empty
    assert "question" in df.columns
    assert len(df) >= 1

@pytest.mark.network
def test_url_csv_loading():
    """Test loading a CSV from a GitHub URL."""
    try:
        import requests
        requests.get("https://raw.githubusercontent.com", timeout=5)
    except Exception:
        pytest.skip("Network is unreachable.")
    
    df = read_csv_flexible(BLOB_URL, cache_dir=None) # Test without caching
    assert not df.empty
    print(f"Loaded CSV with columns: {list(df.columns)}")

def test_url_to_raw_conversion():
    """Test GitHub blob URL to raw URL conversion."""
    from src.utils.dataset_loader import to_raw
    blob_url = "https://github.com/user/repo/blob/main/data/file.csv"
    raw_url = to_raw(blob_url)
    assert raw_url == "https://raw.githubusercontent.com/user/repo/main/data/file.csv"
