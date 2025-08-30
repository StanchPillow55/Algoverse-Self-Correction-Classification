from __future__ import annotations
import os, subprocess, sys, json, tempfile, shutil

class HumanEvalEvaluator:
    """
    Execute candidates against canonical tests (evalplus or humaneval-run).
    Inputs: JSONL with fields {task_id, completion/code, entry_point}.
    """
    def _have_evalplus(self) -> bool:
        try:
            import evalplus  # noqa
            return True
        except Exception:
            return False

    def score(self, samples_path: str, k_list=(1,5), timeout_s=8) -> dict:
        assert os.path.exists(samples_path), samples_path
        if self._have_evalplus():
            cmd = [
                sys.executable, "-m", "evalplus.eval",
                "--samples", samples_path,
                "--k", *map(str, k_list),
                "--timeout", str(timeout_s),
            ]
        else:
            cmd = [
                sys.executable, "-m", "humaneval", "evaluate",
                "--samples", samples_path,
                "--k", *map(str, k_list),
                "--timeout", str(timeout_s),
            ]
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)  # never leak keys to subprocess output
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
        if p.returncode != 0:
            raise RuntimeError(f"HumanEval exec failed: {p.stderr.strip() or p.stdout.strip()}")
        # Expect JSON on stdout with pass@k; fallback: parse last JSON object in stream.
        out = p.stdout.strip().splitlines()
        payload = json.loads(out[-1])
        return {f"pass@{k}": float(payload.get(f"pass@{k}", 0.0)) for k in k_list}
