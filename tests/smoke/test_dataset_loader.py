import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.dataset_loader import read_csv_flexible
URL = "https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification/blob/feat/initial-error-table/data/ground_truth_qna.csv"
def test_local_csv(): assert not read_csv_flexible("data/math20.csv").empty
@pytest.mark.network
def test_url_csv():
    try:
        import requests; requests.get("https://github.com", timeout=5)
    except:
        pytest.skip("Network unavailable")
    assert not read_csv_flexible(URL).empty
