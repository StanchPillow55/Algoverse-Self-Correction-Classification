import csv, json, os
from pathlib import Path
from typing import Dict, Any, List

from src.agents.learner import LearnerBot
from src.agents.teacher import detect_bias, combine_confidence
from src.rts.policy import select_template

def accuracy(answer: str, reference: str) -> int:
    # numeric exact or string exact
    try:
        if "." in answer or "." in reference:
            return int(abs(float(answer) - float(reference)) < 1e-9)
        return int(int(float(answer)) == int(float(reference)))
    except Exception:
        return int((answer or "").strip() == (reference or "").strip())

def run_dataset(
    dataset_csv: str,
    traces_out: str = "outputs/traces.json",
    max_turns: int = 3,
    provider: str = "demo",
    k: int = 1
) -> Dict[str, Any]:
    os.makedirs("outputs", exist_ok=True)

    learner = LearnerBot(provider=provider)
    traces: List[Dict[str, Any]] = []

    with open(dataset_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for idx, row in enumerate(rows):
        qid = f"q{idx+1}"
        q = row["question"]
        ref = str(row["reference"])
        history: List[Dict[str, Any]] = []
        # first attempt
        a0, self_conf = learner.answer(q, history, template=None)
        acc0 = accuracy(a0, ref)
        bias, tconf = detect_bias(q, a0, ref, history)
        conf = combine_confidence(self_conf, tconf, None)

        turns = [{
            "answer": a0, "self_conf": round(self_conf,2), "teacher_bias": bias,
            "teacher_conf": round(tconf,2), "template": None, "accuracy": acc0
        }]

        acc_prev = acc0
        t = 1
        while t < max_turns and acc_prev == 0:
            reprompt, template = select_template(bias, conf, bool(acc_prev), len(history))
            if not reprompt:
                break
            # send template to learner
            a1, self_conf = learner.answer(q, history + turns, template=template)
            acc1 = accuracy(a1, ref)
            bias, tconf = detect_bias(q, a1, ref, history + turns)
            conf = combine_confidence(self_conf, tconf, None)
            turns.append({
                "answer": a1, "self_conf": round(self_conf,2), "teacher_bias": bias,
                "teacher_conf": round(tconf,2), "template": template, "accuracy": acc1
            })
            t += 1
            # simple stop: two non-improvements handled implicitly by max_turns and correctness
            if acc1 == 1: break
            acc_prev = acc1

        traces.append({
            "qid": qid, "question": q, "reference": ref, "turns": turns,
            "final_accuracy": turns[-1]["accuracy"]
        })

    summary = {
        "items": len(traces),
        "final_accuracy_mean": sum(t["final_accuracy"] for t in traces) / max(1,len(traces))
    }
    out = {"summary": summary, "traces": traces}
    with open(traces_out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
