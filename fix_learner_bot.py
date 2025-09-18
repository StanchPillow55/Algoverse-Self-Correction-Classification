#!/usr/bin/env python3
"""
Fix for the learner bot to capture full reasoning traces instead of just final answers.

This script modifies the learner bot to:
1. Return both the full response and the extracted answer
2. Update the tracing system to save complete reasoning traces
3. Ensure the full reasoning process is preserved for analysis
"""

import os
import re
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any

def _first_number(s: str):
    """Extract the first number from a string."""
    m = re.search(r'[-+]?\d+(?:\.\d+)?', s or "")
    return m.group(0) if m else None

class FixedLearnerBot:
    """
    Fixed version of LearnerBot that captures full reasoning traces.
    
    Key changes:
    - Returns both full_response and extracted_answer
    - Preserves complete reasoning process
    - Maintains backward compatibility
    """
    
    def __init__(self, provider: str = "demo", model: str | None = None):
        self.provider = provider
        if model:
            self.model = model
        elif provider == "openai":
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif provider == "anthropic":
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        elif provider == "replicate":
            self.model = os.getenv("REPLICATE_MODEL", "meta/meta-llama-3-70b")
        else:
            self.model = "gpt-4o-mini"  # fallback

    def answer(self, q: str, hist: List[Dict[str, Any]], template: str | None = None, 
               experiment_id: str = "unknown", dataset_name: str = "unknown", 
               sample_id: str = "unknown", turn_number: int = 0) -> Tuple[str, str, float]:
        """
        Returns: (full_response, extracted_answer, confidence)
        
        This is the key fix - we now return the full response along with the extracted answer.
        """
        if os.getenv("DEMO_MODE", "0") == "1" or self.provider == "demo":
            try:
                # Try to isolate the actual question portion if present
                q_text = q
                if "Question:\n" in q:
                    q_text = q.split("Question:\n",1)[1]
                if "Output format:" in q_text:
                    q_text = q_text.split("Output format:",1)[0]
                # Basic arithmetic eval for demo over the question only
                expr = "".join(c for c in q_text if c in "0123456789+-*/. ")
                val = eval(expr)
                answer = str(int(val)) if float(val).is_integer() else f"{val:.2f}"
                full_response = f"I need to solve: {q_text}\nLet me calculate: {expr} = {answer}"
                return full_response, answer, 0.95
            except:
                return "I'm not sure how to solve this problem.", "0", 0.3

        if self.provider == "openai":
            return self._call_openai(q, template, experiment_id, dataset_name, sample_id, turn_number)
        elif self.provider == "anthropic":
            return self._call_anthropic(q, template, experiment_id, dataset_name, sample_id, turn_number)
        elif self.provider == "replicate":
            return self._call_replicate(q, template, experiment_id, dataset_name, sample_id, turn_number)
        else:
            return "UNKNOWN_PROVIDER", "UNKNOWN_PROVIDER", 0.1

    def _call_openai(self, q: str, template: str | None = None, 
                     experiment_id: str = "unknown", dataset_name: str = "unknown", 
                     sample_id: str = "unknown", turn_number: int = 0) -> Tuple[str, str, float]:
        """Call OpenAI API and return full response + extracted answer."""
        from openai import OpenAI
        from ..utils.rate_limit import safe_openai_chat_completion
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Handle template parameter
        user_prompt = f"{q}\n[Instruction]: {template}" if template else q
        
        # Heuristic: detect code-generation tasks
        is_code_task = ("Python function" in q) or ("code block" in q) or ("def " in q)
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1024" if is_code_task else "512"))  # Increased for reasoning
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        
        try:
            resp = safe_openai_chat_completion(
                client=client,
                model=self.model,
                messages=[{"role":"user","content":user_prompt}],
                temperature=temperature, max_tokens=max_tokens
            )
            
            # Extract content
            raw_text = resp.choices[0].message.content
            if raw_text is None:
                return "ERROR: No response received", "ERROR_NULL_RESPONSE", 0.1
                
            full_response = raw_text.strip()
            if not full_response:
                return "ERROR: Empty response", "ERROR_EMPTY_RESPONSE", 0.1
            
            if is_code_task:
                # Extract code from markdown blocks if present
                code_match = re.search(r'```(?:python)?\n?(.*?)\n?```', full_response, re.DOTALL)
                if code_match:
                    extracted_answer = code_match.group(1).strip()
                else:
                    extracted_answer = full_response
                conf = 0.6
            else:
                # Extract numeric answer but keep full response
                num = _first_number(full_response)
                extracted_answer = num if num is not None else full_response[:256]
                conf = 0.85 if (num is not None and num == full_response.strip()) else 0.6
            
            # Track cost and token usage
            try:
                from ..utils.cost_tracker import record_cost
                input_tokens = resp.usage.prompt_tokens if hasattr(resp, 'usage') and resp.usage else 0
                output_tokens = resp.usage.completion_tokens if hasattr(resp, 'usage') and resp.usage else 0
                record_cost(self.model, "openai", input_tokens, output_tokens, 
                           experiment_id, dataset_name, sample_id, turn_number)
            except Exception:
                pass
            
            return full_response, extracted_answer, conf
            
        except Exception as e:
            error_msg = f"API_ERROR: {str(e)}"
            return error_msg, f"ERROR_{type(e).__name__}", 0.1

    def _call_anthropic(self, q: str, template: str | None = None, 
                        experiment_id: str = "unknown", dataset_name: str = "unknown", 
                        sample_id: str = "unknown", turn_number: int = 0) -> Tuple[str, str, float]:
        """Call Anthropic Claude API and return full response + extracted answer."""
        try:
            import anthropic
            from ..utils.rate_limit import safe_anthropic_messages_create
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            
            # Handle template parameter
            user_prompt = f"{q}\n[Instruction]: {template}" if template else q
            
            # Heuristic: detect code-generation tasks
            is_code_task = ("Python function" in q) or ("code block" in q) or ("def " in q)
            max_tokens = 1024 if is_code_task else 512  # Increased for reasoning
            temperature = float(os.getenv("ANTHROPIC_TEMPERATURE", "0.0"))
            
            response = safe_anthropic_messages_create(
                client=client,
                model=self.model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            full_response = response.content[0].text.strip()
            if not full_response:
                return "ERROR: Empty response", "ERROR_EMPTY_RESPONSE", 0.1
            
            if is_code_task:
                # Extract code from markdown blocks if present
                code_match = re.search(r'```(?:python)?\n?(.*?)\n?```', full_response, re.DOTALL)
                if code_match:
                    extracted_answer = code_match.group(1).strip()
                else:
                    extracted_answer = full_response
                conf = 0.6
            else:
                # Extract numeric answer but keep full response
                num = _first_number(full_response)
                extracted_answer = num if num is not None else full_response[:256]
                conf = 0.85 if (num is not None and num == full_response.strip()) else 0.6
            
            return full_response, extracted_answer, conf
            
        except Exception as e:
            error_msg = f"API_ERROR: {str(e)}"
            return error_msg, f"ERROR_{type(e).__name__}", 0.1

    def _call_replicate(self, q: str, template: str | None = None, 
                        experiment_id: str = "unknown", dataset_name: str = "unknown", 
                        sample_id: str = "unknown", turn_number: int = 0) -> Tuple[str, str, float]:
        """Call Replicate API and return full response + extracted answer."""
        try:
            from ..utils.rate_limit import safe_replicate_run
            
            # Handle template parameter
            user_prompt = f"{q}\n[Instruction]: {template}" if template else q
            
            # Heuristic: detect code-generation tasks
            is_code_task = ("Python function" in q) or ("code block" in q) or ("def " in q)
            max_tokens = 1024 if is_code_task else 512  # Increased for reasoning
            temperature = float(os.getenv("REPLICATE_TEMPERATURE", "0.2"))
            
            response = safe_replicate_run(
                self.model,
                input_params={
                    "prompt": user_prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            
            # Replicate returns a generator, so we need to collect the output
            full_response = ""
            for item in response:
                full_response += str(item)
            
            full_response = full_response.strip()
            if not full_response:
                return "ERROR: Empty response", "ERROR_EMPTY_RESPONSE", 0.1
            
            if is_code_task:
                # Extract code from markdown blocks if present
                code_match = re.search(r'```(?:python)?\n?(.*?)\n?```', full_response, re.DOTALL)
                if code_match:
                    extracted_answer = code_match.group(1).strip()
                else:
                    extracted_answer = full_response
                conf = 0.6
            else:
                # Extract numeric answer but keep full response
                num = _first_number(full_response)
                extracted_answer = num if num is not None else full_response[:256]
                conf = 0.85 if (num is not None and num == full_response.strip()) else 0.6
            
            return full_response, extracted_answer, conf
            
        except Exception as e:
            error_msg = f"API_ERROR: {str(e)}"
            return error_msg, f"ERROR_{type(e).__name__}", 0.1

def create_fixed_tracing_system():
    """Create a fixed tracing system that saves full reasoning traces."""
    
    class FixedTracingWriter:
        """Fixed version of TracingWriter that saves full reasoning traces."""
        
        def write_gsm8k_cot(self, run_dir: Path, qid: str, turn_idx: int, full_response: str) -> Path:
            """Save the full reasoning trace instead of just the extracted answer."""
            p = run_dir.joinpath("gsm8k").joinpath(self._slug(qid)).joinpath(f"turn_{turn_idx}").joinpath("cot.txt")
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w', encoding='utf-8') as f:
                f.write(full_response)
            return p
        
        def write_he_code(self, run_dir: Path, task_id: str, turn_idx: int, code_text: str) -> Path:
            """Save the full code response."""
            p = run_dir.joinpath("he").joinpath(self._slug(task_id)).joinpath(f"turn_{turn_idx}").joinpath("code.txt")
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w', encoding='utf-8') as f:
                f.write(code_text)
            return p
        
        def write_prompt(self, run_dir: Path, filename: Path, prompt_text: str) -> Path:
            """Save the prompt template."""
            p = run_dir.joinpath("prompts") / filename
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w', encoding='utf-8') as f:
                f.write(prompt_text)
            return p
        
        def _slug(self, text: str) -> str:
            """Create a safe filename from text."""
            import re
            return re.sub(r'[^\w\-_]', '_', str(text))
    
    return FixedTracingWriter()

def create_fixed_runner():
    """Create a fixed runner that uses the new learner bot and tracing system."""
    
    # This would be a modified version of the runner that:
    # 1. Uses FixedLearnerBot instead of LearnerBot
    # 2. Captures both full_response and extracted_answer
    # 3. Saves full reasoning traces using FixedTracingWriter
    # 4. Maintains backward compatibility with existing metrics
    
    pass

if __name__ == "__main__":
    print("🔧 Learner Bot Fix")
    print("=" * 50)
    print("This script provides the fixes needed to capture full reasoning traces.")
    print("\nKey changes:")
    print("1. FixedLearnerBot returns (full_response, extracted_answer, confidence)")
    print("2. FixedTracingWriter saves complete reasoning traces")
    print("3. Increased max_tokens to capture full reasoning")
    print("4. Maintains backward compatibility")
    print("\nNext steps:")
    print("1. Replace LearnerBot with FixedLearnerBot in the pipeline")
    print("2. Update the runner to handle the new return format")
    print("3. Test with a small subset of experiments")
    print("4. Rerun experiments with the fixed pipeline")
