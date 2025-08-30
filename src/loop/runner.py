from src.utils.dataset_loader import read_csv_flexible
import csv, json, os, yaml
from pathlib import Path
from typing import Dict, Any, List
from src.utils.trace_logger import TraceLogger
from src.metrics.accuracy import gsm8k_em, humaneval_pass_at_k
from src.evaluation.gsm8k_evaluator import GSM8KEvaluator

from src.agents.learner import LearnerBot
from src.agents.teacher import detect_bias, combine_confidence
from src.rts.policy import select_template
from src.evaluator_feedback import coaching_from_bias

# HumanEval support
from src.data.humaneval_loader import load_humaneval_dataset, create_demo_humaneval_data
from src.eval.humaneval_scorer import score_humaneval_candidate

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


def humaneval_accuracy(task: Dict[str, Any], answer: str) -> int:
    """Score HumanEval task using code execution"""
    try:
        score_result = score_humaneval_candidate(task, answer)
        return int(score_result['passed'])
    except Exception as e:
        print(f"Error scoring HumanEval task {task.get('qid', 'unknown')}: {e}")
        return 0


QNA_MAP = {"qid": ["qid","id"], "question": ["question","prompt"], "reference": ["ground_truth","answer"]}
def _auto_map_row(row: dict) -> tuple[str, str, str]:
    low = {str(k).lower().strip(): v for k, v in row.items()}
    def pick(keys): return next((str(low[k.lower()]) for k in keys if k.lower() in low and str(low.get(k.lower(),"")).strip()), "")
    return pick(QNA_MAP["qid"]), pick(QNA_MAP["question"]), pick(QNA_MAP["reference"])


def _load_dataset(dataset_csv: str, subset: str = None) -> List[Dict[str, Any]]:
    """Load dataset, supporting both CSV and HumanEval formats"""
    # Check if this is a HumanEval dataset request
    if dataset_csv == "humaneval" or "humaneval" in dataset_csv.lower():
        try:
            if subset:
                data = load_humaneval_dataset(subset=subset)
            else:
                data = load_humaneval_dataset()
            return data
        except Exception as e:
            print(f"Failed to load HumanEval dataset: {e}")
            print("Using demo HumanEval data...")
            demo_data = create_demo_humaneval_data()
            if subset == "subset_20":
                return demo_data[:20] if len(demo_data) > 20 else demo_data
            elif subset == "subset_100":
                return demo_data[:100] if len(demo_data) > 100 else demo_data
            return demo_data
    else:
        # Traditional CSV loading
        df = read_csv_flexible(dataset_csv)
        return df.to_dict(orient="records")


def run_dataset(
    dataset_csv: str,
    traces_out: str = "outputs/traces.json",
    max_turns: int = 3,
    provider: str = "demo",
    k: int = 1,
    subset: str = None,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    # Ensure output directory exists
    output_dir = Path(traces_out).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    learner = LearnerBot(provider=provider)
    traces: List[Dict[str, Any]] = []

    # Load dataset - supports both CSV and HumanEval formats
    rows = _load_dataset(dataset_csv, subset)
    
    # Initialize trace logger
    run_id = os.environ.get('RUN_ID', 'dev_run')
    split = os.environ.get('DATASET_SPLIT', 'unknown')
    git_sha = os.environ.get('GIT_COMMIT', '')
    logger = TraceLogger(run_id=run_id, dataset_split=split, git_commit=git_sha)
    logger.write_run_config({'dataset': dataset_csv, 'max_turns': max_turns, 'provider': provider, 'split': split, 'model': os.getenv('OPENAI_MODEL')})

    # Apply feature flags from config
    enable_confidence = config.get('features', {}).get('enable_confidence', True) if config else True
    enable_error_awareness = config.get('features', {}).get('enable_error_awareness', True) if config else True
    enable_multi_turn = config.get('features', {}).get('enable_multi_turn', True) if config else True
    
    # Override max_turns if multi-turn is disabled
    if not enable_multi_turn:
        max_turns = 1

    for idx, row in enumerate(rows):
        # Handle different data formats
        if isinstance(row, dict) and row.get('topic') == 'humaneval':
            # HumanEval format with direct fields
            qid = row.get('qid', f"humaneval_{idx}")
            q = row.get('question', '')
            ref = ''
            task = row  # Store the full task for HumanEval scoring
            is_humaneval = True
        else:
            # Traditional CSV format
            qid_m, q, ref = _auto_map_row(row)
            qid = qid_m or f"q{idx+1}"
            task = None
            is_humaneval = False
        
        # Start trace for this example
        ex = logger.start_example(problem_id=qid, text=q)
        
        history: List[Dict[str, Any]] = []
        
        # Load dataset-scoped prompts
        prompts_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "dataset_prompts.yaml")
        try:
            with open(prompts_path, 'r') as f:
                prompt_config = yaml.safe_load(f)
        except Exception:
            # Fallback to hardcoded prompts if file not found
            prompt_config = {
                'humaneval': {
                    'system': 'You are a careful Python programmer. Output only code.',
                    'user_template': 'Implement the following Python function. Return the complete function definition (signature + body) only.\n\nProblem:\n{{problem_text}}\n\nOutput format: Provide a single Python code block containing the full function definition.'
                },
                'gsm8k': {
                    'system': 'You are a meticulous math solver. You may think silently. Provide the final numeric answer.',
                    'user_template': 'Solve the problem. You may include reasoning. End with the final numeric answer on a new line prefixed by "####" or "Answer:".\n\nQuestion:\n{{question}}\n\nOutput format:\n- Full reasoning allowed.\n- Final line MUST be either "#### <number>" or "Answer: <number>".'
                }
            }
        
        # Prepare prompt for HumanEval vs standard tasks
        if is_humaneval:
            pconf = prompt_config.get('humaneval', {})
            # Use template with problem text
            prompt = pconf.get('user_template', '').replace('{{problem_text}}', q).replace('{{function_signature}}', row.get('function_signature', ''))
        else:
            pconf = prompt_config.get('gsm8k', {})
            prompt = pconf.get('user_template', '').replace('{{question}}', q)
        
        # First attempt
        a0, self_conf = learner.answer(prompt, history, template=None)
        
        # Score based on task type
        if is_humaneval:
            # pass@k via sampling (k from env or arg)
            k_try = int(os.getenv("PASS_K", str(k))) if k else int(os.getenv("PASS_K", "1"))
            passes = []
            score_result = score_humaneval_candidate(task, a0)
            execution_details = score_result.get('execution_result', {})
            passes.append(bool(score_result.get('passed', False)))
            for _ in range(max(0, k_try - 1)):
                a_s, _ = learner.answer(prompt, history, template=None)
                sr = score_humaneval_candidate(task, a_s)
                passes.append(bool(sr.get('passed', False)))
            acc0 = int(humaneval_pass_at_k(passes, max(1, k_try)) > 0)
        else:
            # Use enhanced GSM8K evaluator for better extraction and diagnosis
            gsm_eval = GSM8KEvaluator()
            eval_result = gsm_eval.compare(a0, ref)
            acc0 = int(eval_result['em'])
            execution_details = {'diagnosis': eval_result.get('diagnosis', 'unknown')}
        
        # Only run bias detection and confidence if enabled
        if enable_error_awareness:
            bias, tconf = detect_bias(q, a0, ref, history)
        else:
            bias, tconf = "None", 0.5
        
        if enable_confidence:
            conf = combine_confidence(self_conf, tconf, None)
        else:
            conf = 0.5

        turns = [{
            "answer": a0, "self_conf": round(self_conf,2), "teacher_bias": bias,
            "teacher_conf": round(tconf,2), "template": None, "accuracy": acc0,
            "execution_details": execution_details
        }]
        
        # Log first turn
        coaching_feedback = coaching_from_bias(bias)
        logger.on_turn(ex, turn_index=0, prompt=prompt, response_text=a0, 
                      response_is_final=(max_turns == 1 or acc0 == 1), is_correct=bool(acc0),
                      evaluator_signal=('stop' if acc0 == 1 else 'continue'), 
                      model_reported_confidence=self_conf, 
                      evaluator_bias_label=bias,
                      evaluator_feedback=coaching_feedback,
                      model_name=getattr(learner, 'model', provider),
                      task_type='humaneval' if is_humaneval else 'standard',
                      execution_details=execution_details if is_humaneval else None)

        # Multi-turn loop only if enabled (GSM8K only)
        if enable_multi_turn and not is_humaneval:
            acc_prev = acc0
            t = 1
            while t < max_turns and acc_prev == 0:
                # Determine whether to reprompt
                if enable_error_awareness:
                    reprompt, template = select_template(bias, conf, bool(acc_prev), len(history))
                else:
                    # If error awareness is disabled, use a generic retry template
                    reprompt, template = True, "try_again_concise"

                if not reprompt:
                    break

                # Preserve the bias used for template selection to align feedback with the chosen template
                bias_for_template = bias

                # send template to learner
                a1, self_conf = learner.answer(prompt, history + turns, template=template)

                # For GSM8K follow-up turns, use enhanced evaluator
                gsm_eval = GSM8KEvaluator()
                eval_result = gsm_eval.compare(a1, ref)
                acc1 = int(eval_result['em'])
                execution_details = {'diagnosis': eval_result.get('diagnosis', 'unknown')}

                # Only run bias detection and confidence if enabled (after the new response)
                if enable_error_awareness:
                    bias_after, tconf = detect_bias(q, a1, ref, history + turns)
                else:
                    bias_after, tconf = "None", 0.5

                if enable_confidence:
                    conf = combine_confidence(self_conf, tconf, None)
                else:
                    conf = 0.5

                # Append turn details with both before/after bias and selected template
                turns.append({
                    "answer": a1,
                    "self_conf": round(self_conf,2),
                    "teacher_bias": bias_after,
                    "teacher_conf": round(tconf,2),
                    "template": template,
                    "template_selected": template,
                    "evaluator_bias_label_before": bias_for_template,
                    "evaluator_bias_label_after": bias_after,
                    "accuracy": acc1,
                    "execution_details": execution_details
                })

                # Log this turn with feedback aligned to the bias used for template selection
                coaching_feedback = coaching_from_bias(bias_for_template)
                logger.on_turn(
                    ex,
                    turn_index=t,
                    prompt=f"Template: {template}",
                    response_text=a1,
                    response_is_final=(t == max_turns-1 or acc1 == 1),
                    is_correct=bool(acc1),
                    evaluator_signal=('stop' if acc1 == 1 else 'continue'),
                    model_reported_confidence=self_conf,
                    evaluator_bias_label=bias_for_template,
                    evaluator_feedback=coaching_feedback,
                    model_name=getattr(learner, 'model', provider),
                    task_type='humaneval' if is_humaneval else 'standard',
                    template_selected=template,
                    evaluator_bias_label_after=bias_after,
                    execution_details=execution_details if is_humaneval else None
                )

                # Prepare for next iteration: use the latest bias as current
                bias = bias_after

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