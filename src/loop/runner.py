from src.utils.dataset_loader import read_csv_flexible
import csv, json, os
from pathlib import Path
from typing import Dict, Any, List
from src.utils.trace_logger import TraceLogger
from src.metrics.accuracy import gsm8k_em, humaneval_pass_at_k
from src.utils.tracing import RunWriter, RunMeta, sha256_text
from src.utils.ci_bootstrap import mean_ci95
from src.utils.harness_parity import harness_versions
from dataclasses import asdict

from src.agents.learner import LearnerBot
from src.agents.teacher import detect_bias, combine_confidence
from src.rts.policy import select_template
from src.evaluator_feedback import coaching_from_bias

# HumanEval support
from src.data.humaneval_loader import load_humaneval_dataset, create_demo_humaneval_data
from src.eval.humaneval_scorer import score_humaneval_candidate
from src.data.gsm8k_loader import load_gsm8k_dataset

# Enhanced trace formatting and CSV output
from src.utils.enhanced_trace_formatter import EnhancedTraceFormatter
from src.utils.csv_output_formatter import CSVOutputFormatter

# Reasoning extraction and CSV formatting
from src.eval.reasoning_extractor import ReasoningExtractor
from src.eval.csv_formatter import ReasoningCSVFormatter

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
    """Load dataset, supporting both CSV, HumanEval, and GSM8K formats"""
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
    elif dataset_csv == "gsm8k" or "gsm8k" in dataset_csv.lower():
        # Check for GSM8K data
        return load_gsm8k_dataset()
    else:
        # Traditional CSV loading
        df = read_csv_flexible(dataset_csv)
        return df.to_dict(orient="records")


def run_dataset(
    dataset_csv: str,
    traces_out: str = "outputs/traces.json",
    max_turns: int = 3,
    provider: str = "demo",
    model: str = None,
    k: int = 1,
    subset: str = None,
    config: Dict[str, Any] = None,
    experiment_id: str = "unknown",
    dataset_name: str = "unknown"
) -> Dict[str, Any]:
    # Ensure output directory exists
    output_dir = Path(traces_out).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    learner = LearnerBot(provider=provider, model=model)
    traces: List[Dict[str, Any]] = []
    
    # Initialize reasoning extractor and CSV formatter for full reasoning traces
    reasoning_extractor = ReasoningExtractor()
    csv_formatter = ReasoningCSVFormatter(output_dir / "csv_results")

    # Load dataset - supports both CSV and HumanEval formats
    rows = _load_dataset(dataset_csv, subset)
    
    # Initialize trace logger
    run_id = os.environ.get('RUN_ID', 'dev_run')
    split = os.environ.get('DATASET_SPLIT', 'unknown')
    git_sha = os.environ.get('GIT_COMMIT', '')
    logger = TraceLogger(run_id=run_id, dataset_split=split, git_commit=git_sha)
    logger.write_run_config({'dataset': dataset_csv, 'max_turns': max_turns, 'provider': provider, 'split': split, 'model': getattr(learner, 'model', provider)})

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
        sample_id = qid
        ex = logger.start_example(problem_id=qid, text=q)
        
        history: List[Dict[str, Any]] = []
        
        # Prepare prompt for HumanEval vs standard tasks - UPDATED FOR FULL REASONING
        if is_humaneval:
            prompt = (
                "You are a Python programmer. Show your complete reasoning and thought process.\n\n"
                "Think through the problem step by step, explain your approach, then implement the solution.\n"
                "Include your reasoning, then provide the complete function definition.\n\nProblem:\n" + q + "\n\n"
                "Please show your full reasoning process and then provide your implementation."
            )
        else:
            prompt = (
                "You are a math problem solver. Show your complete reasoning and work.\n\n"
                "Think through the problem step by step. Show all calculations and explain your reasoning.\n"
                "Work through the problem completely and provide your final answer.\n\nQuestion:\n" + q + "\n\n"
                "Please show all your work and reasoning, then state your final answer."
            )
        
        # First attempt - Get full reasoning trace
        raw_answer, self_conf, full_response_0 = learner.answer(prompt, history, template=None, 
                                                        experiment_id=experiment_id, dataset_name=dataset_name, 
                                                        sample_id=sample_id, turn_number=0)
        
        # Save full reasoning trace to file
        dataset_type = "code" if is_humaneval else "math"
        reasoning_trace_file = reasoning_extractor.save_reasoning_trace(
            qid=sample_id, turn=0, reasoning_text=full_response_0,
            output_dir=output_dir, dataset_type=dataset_type
        )
        
        # Extract final answer from reasoning trace
        if is_humaneval:
            extracted_answer, reasoning_summary = reasoning_extractor.extract_code_answer(
                full_response_0, task.get('entry_point', '')
            )
        else:
            extracted_answer, reasoning_summary = reasoning_extractor.extract_math_answer(full_response_0)
        
        # Use extracted answer for evaluation, fallback to raw if extraction fails
        a0 = extracted_answer if extracted_answer is not None else raw_answer
        
        # Score based on task type
        if is_humaneval:
            # pass@k via sampling (k from env or arg)
            k_try = int(os.getenv("PASS_K", str(k))) if k else int(os.getenv("PASS_K", "1"))
            passes = []
            score_result = score_humaneval_candidate(task, a0)
            execution_details = score_result.get('execution_result', {})
            passes.append(bool(score_result.get('passed', False)))
            for _ in range(max(0, k_try - 1)):
                # For additional samples, also get full reasoning and extract
                raw_sample, _, full_sample = learner.answer(prompt, history, template=None, 
                                          experiment_id=experiment_id, dataset_name=dataset_name, 
                                          sample_id=sample_id, turn_number=0)
                extracted_sample, _ = reasoning_extractor.extract_code_answer(
                    full_sample, task.get('entry_point', '')
                )
                sample_answer = extracted_sample if extracted_sample is not None else raw_sample
                sr = score_humaneval_candidate(task, sample_answer)
                passes.append(bool(sr.get('passed', False)))
            acc0 = int(humaneval_pass_at_k(passes, max(1, k_try)) > 0)
        else:
            acc0 = gsm8k_em(a0, ref)
            execution_details = {}
        
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
            "answer": a0,  # Extracted answer for evaluation
            "raw_answer": raw_answer,  # Original learner output
            "response_text": full_response_0,  # Full reasoning trace
            "reasoning_trace_file": str(reasoning_trace_file),  # Path to saved reasoning trace
            "reasoning_summary": reasoning_summary,  # Summary of reasoning process
            "self_conf": round(self_conf,2), 
            "teacher_bias": bias,
            "teacher_conf": round(tconf,2),
            "combined_confidence": round(conf,2),
            "template": None, 
            "accuracy": acc0,
            "execution_details": execution_details if is_humaneval else {}
        }]
        
        # Log first turn
        coaching_feedback = coaching_from_bias(bias)
        logger.on_turn(ex, turn_index=0, prompt=prompt, response_text=full_response_0, 
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

                # send template to learner - get full reasoning trace
                raw_answer_1, self_conf, full_response_1 = learner.answer(prompt, history + turns, template=template, 
                                                               experiment_id=experiment_id, dataset_name=dataset_name, 
                                                               sample_id=sample_id, turn_number=t)
                
                # Save reasoning trace for this turn
                reasoning_trace_file_1 = reasoning_extractor.save_reasoning_trace(
                    qid=sample_id, turn=t, reasoning_text=full_response_1,
                    output_dir=output_dir, dataset_type="math"
                )
                
                # Extract answer from reasoning trace
                extracted_answer_1, reasoning_summary_1 = reasoning_extractor.extract_math_answer(full_response_1)
                a1 = extracted_answer_1 if extracted_answer_1 is not None else raw_answer_1

                # For GSM8K follow-up turns, use EM
                acc1 = gsm8k_em(a1, ref)
                execution_details = {}

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
                    "answer": a1,  # Extracted answer for evaluation
                    "raw_answer": raw_answer_1,  # Original learner output
                    "response_text": full_response_1,  # Full reasoning trace
                    "reasoning_trace_file": str(reasoning_trace_file_1),  # Path to saved reasoning trace
                    "reasoning_summary": reasoning_summary_1,  # Summary of reasoning process
                    "self_conf": round(self_conf,2),
                    "teacher_bias": bias_after,
                    "teacher_conf": round(tconf,2),
                    "combined_confidence": round(conf,2),
                    "template": template,
                    "template_selected": template,
                    "evaluator_bias_label_before": bias_for_template,
                    "evaluator_bias_label_after": bias_after,
                    "accuracy": acc1,
                    "execution_details": execution_details if is_humaneval else {}
                })

                # Log this turn with feedback aligned to the bias used for template selection
                coaching_feedback = coaching_from_bias(bias_for_template)
                logger.on_turn(
                    ex,
                    turn_index=t,
                    prompt=f"Template: {template}",
                    response_text=full_response_1,
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
    
    # Generate CSV outputs with reasoning traces
    experiment_config = {
        'dataset_name': dataset_name,
        'model': getattr(learner, 'model', model),
        'provider': provider,
        'temperature': float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        'max_turns': max_turns,
        'experiment_id': experiment_id
    }
    
    print("🔧 Generating CSV outputs with reasoning traces...")
    try:
        results_csv = csv_formatter.format_experiment_results(traces, experiment_config)
        summary_csv = csv_formatter.format_summary_results(traces, experiment_config) 
        turn_analysis_csv = csv_formatter.format_turn_analysis(traces)
        
        from src.eval.csv_formatter import create_analysis_dashboard
        dashboard = create_analysis_dashboard(
            [results_csv, summary_csv, turn_analysis_csv], 
            output_dir / "csv_results"
        )
        
        print(f"✅ CSV analysis complete. Dashboard: {dashboard}")
    except Exception as e:
        print(f"⚠️ CSV generation failed: {e}")

    # New: emit paper-ready run directory with per-turn artifacts and structured traces
    try:
        # Derive run metadata
        seeds_env = os.getenv("SEEDS", "1,2,3")
        seeds = [int(s) for s in seeds_env.split(",") if s.strip().isdigit()]
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        model_name = getattr(learner, 'model', provider)
        arm = os.getenv('RUN_ID', 'dev').lower()
        dataset_name = 'humaneval' if (isinstance(rows[0], dict) and rows[0].get('topic') == 'humaneval') else Path(str(dataset_csv)).stem
        meta = RunMeta(
            arm=arm,
            model=model_name,
            dataset=dataset_name,
            seeds=seeds,
            temperature=temperature,
            max_turns=max_turns,
            harness_versions=harness_versions(),
            start_time=os.getenv('RUN_START', ''),
            end_time=os.getenv('RUN_END', ''),
            git_commit=os.getenv('GIT_COMMIT', '')
        )
        writer = RunWriter(Path('runs'))
        # Choose first seed for directory naming to keep deterministic single-run output
        run_dir = writer.make_run_dir(meta, seed=seeds[0] if seeds else 1)

        # Save config snapshot
        cfg_snapshot = {
            'dataset': dataset_csv,
            'subset': subset,
            'max_turns': max_turns,
            'provider': provider,
            'model': model_name,
            'seeds': seeds,
            'temperature': temperature,
            'meta': {
                'git_commit': os.getenv('GIT_COMMIT', ''),
                'harness_versions': harness_versions(),
                'os': os.uname().sysname if hasattr(os, 'uname') else 'N/A',
            }
        }
        writer.write_config(run_dir, cfg_snapshot)

        # Build paper traces schema
        items = []
        for ex in traces:
            is_he = 'topic' in ex.get('question','').lower() or (isinstance(rows[0], dict) and rows[0].get('topic')=='humaneval')
            turns_schema = []
            for ti, t in enumerate(ex['turns']):
                # Write per-turn artifacts
                if is_he:
                    ref_path = writer.write_he_code(run_dir, ex['qid'], ti, t.get('answer',''))
                else:
                    # Try to get the actual response content from the turn data
                    response_content = t.get('response_text', '') or t.get('answer', '') or t.get('normalized_answer', '') or ''
                    ref_path = writer.write_gsm8k_cot(run_dir, ex['qid'], ti, response_content)
                # Write prompt template record
                prompt_text = t.get('template') or ''
                prompt_ref = None
                if prompt_text:
                    prompt_ref = writer.write_prompt(run_dir, Path(f"turn_prompt_{ti}_{sha256_text(ex['qid'])[:8]}.txt"), prompt_text)
                turns_schema.append({
                    'turn_index': ti,
                    'prompt_id': t.get('template') or 'base',
                    'prompt_text_ref': str(prompt_ref) if prompt_ref else '',
                    'output_sha256': sha256_text(t.get('answer','')),
                    'learner_output_ref': str(ref_path),
                    'evaluator_feedback': t.get('execution_details', {}),
                    'confidence': t.get('self_conf', None),
                    'normalized_answer': None,
                    'exec_result': t.get('execution_details', {}) if is_he else None,
                    'accuracy': t.get('accuracy', 0),  # Add accuracy field
                    'response_text': t.get('response_text', ''),  # Add response_text field
                })
            items.append({
                'id': ex['qid'],
                'turns': turns_schema,
                'final': {
                    'predicted': ex['turns'][-1].get('answer',''),
                    'correct': bool(ex['final_accuracy']),
                    'error_taxonomy': None
                }
            })

        # Metrics with CI
        accs = [float(x.get('final_accuracy',0)) for x in traces]
        mean, lo, hi = mean_ci95(accs, reps=1000, rng_seed=0)
        metrics = {
            'accuracy_mean': mean,
            'ci95': [lo, hi],
            'n': len(accs),
            'seeds': seeds,
        }
        writer.write_metrics(run_dir, metrics)
        writer.write_traces(run_dir, {'meta': asdict(meta), 'items': items})
        
        # NEW: Generate structured traces in desired format
        try:
            print("🔧 Generating structured trace outputs...")
            
            # Generate individual problem trace files
            structured_files = []
            for trace in traces:
                problem_file = writer.write_structured_problem_trace(
                    run_dir, trace['qid'], trace['turns']
                )
                structured_files.append(str(problem_file))
                
            # Generate flat JSON results with automatic dataset type detection
            flat_json_file = writer.write_flat_json_results(run_dir, traces, "unknown")
            
            print(f"✅ Structured traces generated:")
            print(f"  - Problem trace files: {len(structured_files)} files")
            print(f"  - Flat JSON results: {flat_json_file}")
            
            # Add structured trace info to output
            out['structured_traces'] = {
                'problem_files': structured_files,
                'flat_json': str(flat_json_file),
                'format': 'desired_paper_format'
            }
            
        except Exception as e:
            print(f"⚠️ Structured trace generation failed: {e}")
            # Don't fail the run if structured traces fail
    except Exception as _e:
        # Do not fail the run if writer errors; proceed silently (logged to stdout)
        print(f"WARN: tracing writer failed: {_e}")

    # Close the trace logger
    logger.close()
    
    # Enhanced trace formatting and CSV output
    try:
        print("🔧 Creating enhanced trace formatting and CSV outputs...")
        
        # Create enhanced trace formatter
        enhanced_formatter = EnhancedTraceFormatter()
        traces_file = str(run_dir / "traces.json")
        
        if Path(traces_file).exists():
            # Format enhanced traces
            enhanced_outputs = enhanced_formatter.format_experiment_traces(traces_file, experiment_id)
            print(f"✅ Enhanced traces created: {len(enhanced_outputs)} outputs")
            
            # Create CSV outputs
            csv_formatter = CSVOutputFormatter()
            final_answers_csv = csv_formatter.create_final_answers_csv(traces_file, experiment_id)
            multi_turn_csv = csv_formatter.create_multi_turn_accuracy_csv(traces_file, experiment_id)
            summary_csv = csv_formatter.create_summary_metrics_csv(traces_file, experiment_id)
            
            print(f"✅ CSV outputs created:")
            print(f"  - Final answers: {final_answers_csv}")
            print(f"  - Multi-turn accuracy: {multi_turn_csv}")
            print(f"  - Summary metrics: {summary_csv}")
            
            # Add CSV paths to output
            out['enhanced_traces'] = enhanced_outputs
            out['csv_outputs'] = {
                'final_answers': final_answers_csv,
                'multi_turn_accuracy': multi_turn_csv,
                'summary_metrics': summary_csv
            }
        else:
            print("⚠️ No traces.jsonl found for enhanced formatting")
            
    except Exception as e:
        print(f"⚠️ Enhanced formatting failed: {e}")
        # Don't fail the run if formatting fails
    
    return out
