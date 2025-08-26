from src.utils.dataset_loader import read_csv_flexible
import csv, json, os
from pathlib import Path
from typing import Dict, Any, List
from src.utils.trace_logger import TraceLogger

from src.agents.learner import LearnerBot
from src.agents.teacher import detect_bias, combine_confidence
from src.rts.policy import select_template
from src.evaluator_feedback import coaching_from_bias

mismatch_log = 'outputs/mismatches.log'

def accuracy(answer: str, reference: str) -> int:
    # numeric exact or string exact
    try:
        if "." in answer or "." in reference:
            return int(abs(float(answer) - float(reference)) < 1e-9)
        return int(int(float(answer)) == int(float(reference)))
    except Exception:
        ans = (answer or "").strip()
    ref = (reference or "").strip()
    ok = int(ans == ref)
    if not ok:
        with open(mismatch_log, 'a', encoding='utf-8') as f:
            f.write(f'MISMATCH | Parsed Answer: "{ans}" | Expected Reference: "{ref}"\n')
    return ok


QNA_MAP = {"qid": ["qid","id"], "question": ["question","prompt"], "reference": ["ground_truth","answer"]}
def _auto_map_row(row: dict) -> tuple[str, str, str]:
    low = {str(k).lower().strip(): v for k, v in row.items()}
    def pick(keys): return next((str(low[k.lower()]) for k in keys if k.lower() in low and str(low.get(k.lower(),"")).strip()), "")
    return pick(QNA_MAP["qid"]), pick(QNA_MAP["question"]), pick(QNA_MAP["reference"])


def run_dataset(
    dataset_csv: str,
    traces_out: str = "outputs/traces.json",
    max_turns: int = 3,
    provider: str = "demo",
    k: int = 1
) -> Dict[str, Any]:
    # Ensure output directory exists
    output_dir = Path(traces_out).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    learner = LearnerBot(provider=provider)
    traces: List[Dict[str, Any]] = []

    df = read_csv_flexible(dataset_csv)
    rows = df.to_dict(orient="records")
    
    # Initialize trace logger
    run_id = os.environ.get('RUN_ID', 'dev_run')
    split = os.environ.get('DATASET_SPLIT', 'unknown')
    git_sha = os.environ.get('GIT_COMMIT', '')
    logger = TraceLogger(run_id=run_id, dataset_split=split, git_commit=git_sha)
    logger.write_run_config({'dataset': dataset_csv, 'max_turns': max_turns, 'provider': provider, 'split': split, 'model': os.getenv('OPENAI_MODEL')})

    for idx, row in enumerate(rows):
        qid_m, q, ref = _auto_map_row(row)
        qid = qid_m or f"q{idx+1}"
        
        # Start trace for this example
        ex = logger.start_example(problem_id=qid, text=q)
        
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
        
        # Log first turn
        coaching_feedback = coaching_from_bias(bias)
        logger.on_turn(ex, turn_index=0, prompt=q, response_text=a0, 
                      response_is_final=(max_turns == 1 or acc0 == 1), is_correct=bool(acc0),
                      evaluator_signal=('stop' if acc0 == 1 else 'continue'), 
                      model_reported_confidence=self_conf, 
                      evaluator_bias_label=bias,
                      evaluator_feedback=coaching_feedback,
                      model_name=getattr(learner, 'model', provider))

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
            
            # Log this turn
            coaching_feedback = coaching_from_bias(bias)
            logger.on_turn(ex, turn_index=t, prompt=f"Template: {template}", response_text=a1, 
                          response_is_final=(t == max_turns-1 or acc1 == 1), is_correct=bool(acc1),
                          evaluator_signal=('stop' if acc1 == 1 else 'continue'), 
                          model_reported_confidence=self_conf, 
                          evaluator_bias_label=bias,
                          evaluator_feedback=coaching_feedback,
                          model_name=getattr(learner, 'model', provider))
            
            t += 1
            # simple stop: two non-improvements handled implicitly by max_turns and correctness
            if acc1 == 1: break
            acc_prev = acc1

        # Finalize the trace for this example
        final_answer = turns[-1]["answer"]
        final_correct = bool(turns[-1]["accuracy"])
        logger.end_example(ex, final_answer=final_answer, final_correct=final_correct)
        
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
    
    # Close the trace logger
    logger.close()
    return out
