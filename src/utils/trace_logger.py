import os, json, time, datetime as dt
from typing import Optional, Dict, Any

class TraceLogger:
    """
    Full-schema trace logger that captures comprehensive turn-by-turn data:
    - Prompts, responses, feedback, confidence, signals, correctness, latency
    - Problem metadata and final results
    - Complete multi-turn conversation traces
    """
    def __init__(self, run_id: str, out_dir: str = "./runs", dataset_split: str = "unknown", git_commit: str = ""):
        self.run_id = run_id
        self.dataset_split = dataset_split
        self.git_commit = git_commit
        self.root = os.path.join(out_dir, run_id)
        os.makedirs(self.root, exist_ok=True)
        self.path = os.path.join(self.root, "traces.jsonl")
        self._fh = open(self.path, "a", encoding="utf-8")

    def close(self):
        if self._fh: 
            self._fh.close()
            self._fh = None

    def start_example(self, problem_id: str, text: str) -> Dict[str, Any]:
        """Start tracking a new problem example with full metadata"""
        return {
            "problem_id": str(problem_id), 
            "dataset_split": self.dataset_split,
            "original_problem_text": text, 
            "turns": [], 
            "final_answer": "",
            "final_correct": None, 
            "num_turns": 0, 
            "run_id": self.run_id,
            "git_commit": self.git_commit, 
            "time_started": dt.datetime.utcnow().isoformat()+"Z"
        }

    def on_turn(self, ex: Dict[str, Any], **kwargs):
        """Log a complete turn with all available data"""
        # Capture comprehensive turn details including:
        # - prompts, responses, feedback, confidence, signals, correctness, latency
        turn_data = {k: v for k, v in kwargs.items()}
        
        # Add timestamp for this turn
        turn_data["turn_timestamp"] = dt.datetime.utcnow().isoformat()+"Z"
        
        ex["turns"].append(turn_data)

    def end_example(self, ex: Dict[str, Any], final_answer: str, final_correct: bool):
        """Complete the example trace with final results"""
        ex.update({
            "final_answer": str(final_answer), 
            "final_correct": bool(final_correct),
            "num_turns": len(ex["turns"]), 
            "time_finished": dt.datetime.utcnow().isoformat()+"Z"
        })
        self._fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        self._fh.flush()

    def write_run_config(self, cfg: dict):
        """Save run configuration metadata"""
        with open(os.path.join(self.root, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
