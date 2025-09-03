#!/usr/bin/env python3
import json, os, hashlib, tempfile, shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def safe_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-","_") else "-" for c in s.lower()).strip("-")


@dataclass
class RunMeta:
    arm: str
    model: str
    dataset: str
    seeds: List[int]
    temperature: float
    max_turns: int
    harness_versions: Dict[str, str]
    start_time: str
    end_time: Optional[str]
    git_commit: str
    tokenizer_version: Optional[str] = None


class RunWriter:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def make_run_dir(self, meta: RunMeta, seed: int) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dname = f"{date_str}__{_slug(meta.dataset)}__{_slug(meta.arm)}__{_slug(meta.model)}__seed{seed}__t{meta.temperature}__mt{meta.max_turns}"
        return self.base_dir / dname

    def write_config(self, run_dir: Path, cfg: Dict[str, Any]) -> None:
        safe_write(run_dir.joinpath("config.json"), json.dumps(cfg, indent=2).encode("utf-8"))

    def write_metrics(self, run_dir: Path, metrics: Dict[str, Any]) -> None:
        safe_write(run_dir.joinpath("metrics.json"), json.dumps(metrics, indent=2).encode("utf-8"))

    def write_traces(self, run_dir: Path, traces: Dict[str, Any]) -> None:
        safe_write(run_dir.joinpath("traces.json"), json.dumps(traces, indent=2).encode("utf-8"))

    def write_prompt(self, run_dir: Path, rel_path: Path, content: str) -> Path:
        p = run_dir.joinpath("prompts").joinpath(rel_path)
        safe_write(p, content.encode("utf-8"))
        return p

    def write_he_code(self, run_dir: Path, task_id: str, turn_idx: int, code_text: str) -> Path:
        p = run_dir.joinpath("he").joinpath(_slug(task_id)).joinpath(f"turn_{turn_idx}").joinpath("code.txt")
        safe_write(p, code_text.encode("utf-8"))
        return p

    def write_gsm8k_cot(self, run_dir: Path, qid: str, turn_idx: int, cot_text: str) -> Path:
        p = run_dir.joinpath("gsm8k").joinpath(_slug(qid)).joinpath(f"turn_{turn_idx}").joinpath("cot.txt")
        safe_write(p, cot_text.encode("utf-8"))
        return p


