from src.utils.harness_parity import harness_versions

def test_harness_versions_has_required_keys():
    hv = harness_versions()
    for k in ("evalplus_version", "humaneval_harness", "gsm8k_extractor_version"):
        assert k in hv
