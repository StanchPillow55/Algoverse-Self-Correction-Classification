# Legacy Tree (Pre-Pivot)

This folder stores the pre-pivot classification/embeddings pipelines, tests, and datasets, mirrored under `legacy/` with their original structure (e.g., `legacy/src/classification`, `legacy/tests/unit`, `legacy/data`).

They are not executed by default and are excluded from pytest via `norecursedirs = legacy`.

If you need to run any of the legacy components, do so in an isolated virtual environment and do not mix `.env` files with the main pipeline.
