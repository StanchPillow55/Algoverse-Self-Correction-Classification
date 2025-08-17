import os, json, datetime as dt
from typing import Optional, Dict, Any

class TraceLogger:
    def __init__(self, run_id: str, out_dir: str = "./runs", dataset_split: str = "unknown", git_commit: str = ""):
        self.run_id = run_id
        self.dataset_split = dataset_split
        self.git_commit = git_commit
        self.root = os.path.join(out_dir, run_id)
        os.makedirs(self.root, exist_ok=True)
        self.path = os.path.join(self.root, "traces.jsonl")
        self._fh = open(self.path, "a", encoding="utf-8")

    def close(self):
        if self._fh: self._fh.close()

    def start_example(self, problem_id: str, text: str) -> Dict[str, Any]:
        return {"problem_id": str(problem_id), "dataset_split": self.dataset_split,
                "original_problem_text": text, "turns": [], "final_answer": "",
                "final_correct": None, "num_turns": 0, "run_id": self.run_id,
                "git_commit": self.git_commit, "time_started": dt.datetime.utcnow().isoformat()+"Z"}

    def on_turn(self, ex: Dict[str, Any], **kwargs):
        ex["turns"].append({k: v for k, v in kwargs.items()})

    def end_example(self, ex: Dict[str, Any], final_answer: str, final_correct: bool):
        ex.update({"final_answer": str(final_answer), "final_correct": bool(final_correct),
                   "num_turns": len(ex["turns"]), "time_finished": dt.datetime.utcnow().isoformat()+"Z"})
        self._fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        self._fh.flush()

    def write_run_config(self, cfg: dict):
        with open(os.path.join(self.root, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
