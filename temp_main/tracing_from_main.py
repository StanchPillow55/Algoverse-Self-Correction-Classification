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

    def write_structured_problem_trace(self, run_dir: Path, problem_id: str, turns_data: List[Dict[str, Any]]) -> Path:
        """Write problem trace in the desired structured format:
        problem_1.txt
        turn 1
        Reasoning trace1
        Feedback1
        turn2
        Reasoning trace2
        Feedback2
        """
        content_lines = []
        
        for i, turn in enumerate(turns_data, 1):
            content_lines.append(f"turn {i}")
            
            # Add reasoning trace (full response text)
            reasoning = turn.get('response_text', turn.get('answer', ''))
            if reasoning:
                content_lines.append(reasoning)
            
            # Add feedback if available
            feedback = self._extract_feedback_text(turn)
            if feedback:
                content_lines.append(feedback)
                
        content = '\n'.join(content_lines)
        p = run_dir.joinpath("structured_traces").joinpath(f"problem_{problem_id}.txt")
        safe_write(p, content.encode("utf-8"))
        return p

    def _extract_feedback_text(self, turn: Dict[str, Any]) -> str:
        """Extract readable feedback text from turn data"""
        feedback_parts = []
        
        # Check for execution feedback (HumanEval)
        exec_details = turn.get('execution_details', {})
        if exec_details:
            if exec_details.get('passed'):
                feedback_parts.append(f"PASS: All tests passed")
            else:
                feedback_parts.append(f"FAIL: {exec_details.get('error', 'Tests failed')}")
        
        # Check for teacher bias feedback
        teacher_bias = turn.get('teacher_bias')
        if teacher_bias and teacher_bias != 'None':
            feedback_parts.append(f"Bias detected: {teacher_bias}")
            
        # Check for template feedback
        template = turn.get('template')
        if template:
            feedback_parts.append(f"Template applied: {template}")
            
        return '\n'.join(feedback_parts)

    def _detect_dataset_type(self, traces: List[Dict[str, Any]], current_type: str = "unknown") -> str:
        """Detect dataset type from trace characteristics"""
        if current_type != "unknown":
            return current_type
            
        if not traces:
            return "math"  # Default fallback
            
        first_trace = traces[0]
        first_turn = first_trace.get('turns', [{}])[0] if first_trace.get('turns') else {}
        
        # Check question ID patterns and characteristics
        qid = first_trace.get('qid', '').lower()
        question = first_trace.get('question', '').lower()
        
        # HumanEval: has execution details and specific ID pattern
        if (first_turn.get('execution_details') and 
            isinstance(first_turn.get('execution_details'), dict) and
            ('humaneval' in qid or 'eval/' in qid)):
            return "humaneval"
        
        # ToolQA: contains tool-related keywords or patterns
        if ('toolqa' in qid or 
            'tool' in question or 
            'population' in question or 
            'weather' in question or 
            'search' in question):
            return "toolqa"
        
        # SuperGLUE: contains task type indicators or specific patterns
        if ('boolq' in qid or 'copa' in qid or 'rte' in qid or 'wic' in qid or 'wsc' in qid or 'cb' in qid or
            'multirc' in qid or 'superglue' in qid or
            any(task in question for task in ['true or false', 'entailment', 'choice'])):
            return "superglue"
        
        # Default to math for GSM8K/MathBench
        return "math"

    def _extract_tools_used(self, turn: Dict[str, Any]) -> str:
        """Extract tools used from turn data for ToolQA format"""
        # Check various fields that might contain tool information
        tools_used = []
        
        # Check for tools in execution details or other fields
        exec_details = turn.get('execution_details', {})
        if isinstance(exec_details, dict):
            if 'tools_used' in exec_details:
                tools_used.extend(exec_details['tools_used'])
            elif 'tools' in exec_details:
                tools_used.extend(exec_details['tools'])
        
        # Check response text for tool mentions
        response_text = turn.get('response_text', '')
        tool_patterns = ['location', 'weather', 'search', 'calculator', 'database', 'api']
        for tool in tool_patterns:
            if tool in response_text.lower():
                tools_used.append(tool)
        
        # Remove duplicates and return as comma-separated string
        return ','.join(list(set(tools_used))) if tools_used else 'none'

    def _calculate_tool_accuracy(self, turn: Dict[str, Any]) -> float:
        """Calculate tool usage accuracy for ToolQA format"""
        # For now, use the general accuracy as a proxy
        # In a real implementation, this would check tool usage success rate
        accuracy = turn.get('accuracy', 0)
        confidence = turn.get('self_conf', 0.5)
        
        # Combine accuracy and confidence for tool accuracy estimate
        if accuracy == 1:
            return min(1.0, confidence + 0.1)  # Boost for correct answers
        else:
            return max(0.0, confidence - 0.2)  # Reduce for incorrect answers

    def _extract_task_type(self, trace: Dict[str, Any], turn: Dict[str, Any]) -> str:
        """Extract task type for SuperGLUE format"""
        qid = trace.get('qid', '').lower()
        question = trace.get('question', '').lower()
        
        # Map based on question ID patterns
        if 'boolq' in qid or any(word in question for word in ['true', 'false', 'yes', 'no']):
            return 'BoolQ'
        elif 'copa' in qid or 'choice' in question or 'because' in question:
            return 'COPA'
        elif 'rte' in qid or 'entailment' in question or 'follows' in question:
            return 'RTE'
        elif 'wic' in qid or 'same meaning' in question:
            return 'WiC'
        elif 'wsc' in qid or 'pronoun' in question:
            return 'WSC'
        elif 'cb' in qid or 'commitment' in question:
            return 'CB'
        elif 'multirc' in qid or 'multiple' in question:
            return 'MultiRC'
        else:
            return 'Unknown'

    def write_flat_json_results(self, run_dir: Path, traces: List[Dict[str, Any]], dataset_type: str = "unknown") -> Path:
        """Write results in chart/spreadsheet format JSON.
        
        For GSM8K/MathBench (numeric answers):
        |question|turn1finalAns|bias1|feedback1|accuracy1|confidence1|turn2finalAns|...
        
        For HumanEval (code generation):
        |question|turn1finalAcc|bias1|feedback1|testAccuracy1|confidence1|turn2finalAcc|...
        
        For ToolQA (tool reasoning):
        |question|turn1finalAns|bias1|feedback1|toolsUsed1|toolAccuracy1|confidence1|turn2finalAns|...
        
        For SuperGLUE (multiple reasoning tasks):
        |question|turn1finalAns|bias1|feedback1|taskType1|accuracy1|confidence1|turn2finalAns|...
        """
        # Detect dataset type if not provided
        dataset_type = self._detect_dataset_type(traces, dataset_type)
        
        # Create the chart format with columns as keys
        chart_data = {
            "columns": [],
            "rows": []
        }
        
        # Determine max turns across all problems
        max_turns = max(len(trace.get('turns', [])) for trace in traces) if traces else 0
        
        # Build column headers based on dataset type
        if dataset_type == "humaneval":
            # HumanEval: question|turn1finalAcc|bias1|feedback1|testAccuracy1|confidence1|...
            chart_data["columns"] = ["question"]
            for i in range(1, max_turns + 1):
                chart_data["columns"].extend([
                    f"turn{i}finalAcc",
                    f"bias{i}", 
                    f"feedback{i}",
                    f"testAccuracy{i}",
                    f"confidence{i}"
                ])
        elif dataset_type == "toolqa":
            # ToolQA: question|turn1finalAns|bias1|feedback1|toolsUsed1|toolAccuracy1|confidence1|...
            chart_data["columns"] = ["question"]
            for i in range(1, max_turns + 1):
                chart_data["columns"].extend([
                    f"turn{i}finalAns",
                    f"bias{i}",
                    f"feedback{i}",
                    f"toolsUsed{i}",
                    f"toolAccuracy{i}",
                    f"confidence{i}"
                ])
        elif dataset_type == "superglue":
            # SuperGLUE: question|turn1finalAns|bias1|feedback1|taskType1|accuracy1|confidence1|...
            chart_data["columns"] = ["question"]
            for i in range(1, max_turns + 1):
                chart_data["columns"].extend([
                    f"turn{i}finalAns",
                    f"bias{i}",
                    f"feedback{i}",
                    f"taskType{i}",
                    f"accuracy{i}",
                    f"confidence{i}"
                ])
        else:
            # GSM8K/MathBench: question|turn1finalAns|bias1|feedback1|accuracy1|confidence1|...
            chart_data["columns"] = ["question"]
            for i in range(1, max_turns + 1):
                chart_data["columns"].extend([
                    f"turn{i}finalAns",
                    f"bias{i}",
                    f"feedback{i}", 
                    f"accuracy{i}",
                    f"confidence{i}"
                ])
        
        # Build rows (each problem becomes a row)
        for trace in traces:
            row = [trace.get('qid', '')]  # Start with question ID
            
            turns = trace.get('turns', [])
            for i in range(1, max_turns + 1):
                if i <= len(turns):
                    turn = turns[i-1]
                    
                    if dataset_type == "humaneval":
                        # HumanEval format
                        exec_details = turn.get('execution_details', {})
                        passed = exec_details.get('passed', False)
                        passed_count = exec_details.get('passed_count', 0)
                        total_count = exec_details.get('total_count', 0)
                        test_accuracy = passed_count / total_count if total_count > 0 else 0.0
                        
                        row.extend([
                            1 if passed else 0,  # turn{i}finalAcc (0 or 1)
                            turn.get('teacher_bias', 'None'),  # bias{i}
                            self._extract_feedback_text(turn),  # feedback{i}
                            round(test_accuracy, 4),  # testAccuracy{i} (proportion)
                            round(turn.get('self_conf', 0.0), 3)  # confidence{i}
                        ])
                    elif dataset_type == "toolqa":
                        # ToolQA format
                        tools_used = self._extract_tools_used(turn)
                        tool_accuracy = self._calculate_tool_accuracy(turn)
                        
                        row.extend([
                            turn.get('answer', ''),  # turn{i}finalAns
                            turn.get('teacher_bias', 'None'),  # bias{i}
                            self._extract_feedback_text(turn),  # feedback{i}
                            tools_used,  # toolsUsed{i}
                            round(tool_accuracy, 4),  # toolAccuracy{i} (proportion)
                            round(turn.get('self_conf', 0.0), 3)  # confidence{i}
                        ])
                    elif dataset_type == "superglue":
                        # SuperGLUE format
                        task_type = self._extract_task_type(trace, turn)
                        
                        row.extend([
                            turn.get('answer', ''),  # turn{i}finalAns
                            turn.get('teacher_bias', 'None'),  # bias{i}
                            self._extract_feedback_text(turn),  # feedback{i}
                            task_type,  # taskType{i}
                            turn.get('accuracy', 0),  # accuracy{i}
                            round(turn.get('self_conf', 0.0), 3)  # confidence{i}
                        ])
                    else:
                        # GSM8K/MathBench format
                        row.extend([
                            turn.get('answer', ''),  # turn{i}finalAns
                            turn.get('teacher_bias', 'None'),  # bias{i}
                            self._extract_feedback_text(turn),  # feedback{i}
                            turn.get('accuracy', 0),  # accuracy{i}
                            round(turn.get('self_conf', 0.0), 3)  # confidence{i}
                        ])
                else:
                    # Fill empty turns with blanks based on dataset type
                    if dataset_type == "humaneval":
                        row.extend(['', '', '', '', ''])  # finalAcc, bias, feedback, testAccuracy, confidence
                    elif dataset_type == "toolqa":
                        row.extend(['', '', '', '', '', ''])  # finalAns, bias, feedback, toolsUsed, toolAccuracy, confidence
                    elif dataset_type == "superglue":
                        row.extend(['', '', '', '', '', ''])  # finalAns, bias, feedback, taskType, accuracy, confidence  
                    else:
                        row.extend(['', '', '', '', ''])  # finalAns, bias, feedback, accuracy, confidence
            
            chart_data["rows"].append(row)
        
        p = run_dir.joinpath("structured_traces").joinpath("flat_results.json")
        safe_write(p, json.dumps(chart_data, indent=2).encode("utf-8"))
        return p


