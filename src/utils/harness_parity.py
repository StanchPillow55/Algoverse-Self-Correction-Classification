#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict


def harness_versions() -> Dict[str, str]:
    """Return pinned harness versions and notes."""
    return {
        "evalplus_version": "N/A",
        "humaneval_harness": "local_sandbox_v1",
        "gsm8k_extractor_version": "metrics/accuracy.py@normalize_v2",
    }

