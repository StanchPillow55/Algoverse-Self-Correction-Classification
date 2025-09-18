import os, re, json
from typing import Tuple, List, Dict, Any
from pathlib import Path

def _first_number(s: str):
    m = re.search(r'[-+]?\d+(?:\.\d+)?', s or "")
    return m.group(0) if m else None

class LearnerBot:
    def __init__(self, provider: str = "demo", model: str | None = None):
        self.provider = provider
        # self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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
        
        FIXED: Now returns the full reasoning trace along with the extracted answer.
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
        """Call OpenAI API."""
        from openai import OpenAI
        from ..utils.rate_limit import safe_openai_chat_completion
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Avoid global terse constraints; let dataset-level prompt dictate behavior
        sys_prompt = ""
        
        # Handle template parameter
        user_prompt = f"{q}\n[Instruction]: {template}" if template else q
        
        # Heuristic: detect code-generation tasks (HumanEval) to allow larger outputs and avoid numeric-only parsing
        is_code_task = ("Python function" in q) or ("code block" in q) or ("def " in q)
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1024" if is_code_task else "512"))  # Increased for reasoning
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        
        try:
            resp = safe_openai_chat_completion(
                client=client,
                model=self.model,
                messages=(([{"role":"system","content":sys_prompt}] if sys_prompt else []) + [{"role":"user","content":user_prompt}]),
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
            self._safe_debug_log(q, template, full_response, extracted_answer)
            
            # Track cost and token usage
            try:
                from ..utils.cost_tracker import record_cost
                input_tokens = resp.usage.prompt_tokens if hasattr(resp, 'usage') and resp.usage else 0
                output_tokens = resp.usage.completion_tokens if hasattr(resp, 'usage') and resp.usage else 0
                record_cost(self.model, "openai", input_tokens, output_tokens, 
                           experiment_id, dataset_name, sample_id, turn_number)
            except Exception as e:
                # Don't fail the main operation if cost tracking fails
                pass
            
            return full_response, extracted_answer, conf
            
        except Exception as e:
            # FIX: Don't silently fallback to "0" - log the actual error
            error_msg = f"API_ERROR: {str(e)}"
            self._safe_debug_log(q, template, error_msg, "ERROR")
            return error_msg, f"ERROR_{type(e).__name__}", 0.1

    def _call_anthropic(self, q: str, template: str | None = None, 
                        experiment_id: str = "unknown", dataset_name: str = "unknown", 
                        sample_id: str = "unknown", turn_number: int = 0) -> Tuple[str, str, float]:
        """Call Anthropic Claude API."""
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
            
            self._safe_debug_log(q, template, full_response, extracted_answer)
            
            # Track cost and token usage
            try:
                from ..utils.cost_tracker import record_cost
                input_tokens = response.usage.input_tokens if hasattr(response, 'usage') and response.usage else 0
                output_tokens = response.usage.output_tokens if hasattr(response, 'usage') and response.usage else 0
                record_cost(self.model, "anthropic", input_tokens, output_tokens, 
                           experiment_id, dataset_name, sample_id, turn_number)
            except Exception as e:
                # Don't fail the main operation if cost tracking fails
                pass
            
            return full_response, extracted_answer, conf
            
        except Exception as e:
            error_msg = f"ANTHROPIC_API_ERROR: {str(e)}"
            self._safe_debug_log(q, template, error_msg, "ERROR")
            return error_msg, f"ERROR_{type(e).__name__}", 0.1

    def _call_replicate(self, q: str, template: str | None = None, 
                        experiment_id: str = "unknown", dataset_name: str = "unknown", 
                        sample_id: str = "unknown", turn_number: int = 0) -> Tuple[str, str, float]:
        """Call Replicate API (for Llama models)."""
        try:
            from ..utils.rate_limit import safe_replicate_run
            
            # Handle template parameter
            user_prompt = f"{q}\n[Instruction]: {template}" if template else q
            
            # Heuristic: detect code-generation tasks
            is_code_task = ("Python function" in q) or ("code block" in q) or ("def " in q)
            max_tokens = 1024 if is_code_task else 512  # Increased for reasoning
            temperature = float(os.getenv("REPLICATE_TEMPERATURE", "0.0"))
            
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
            
            self._safe_debug_log(q, template, full_response, extracted_answer)
            
            # Track cost and token usage (Replicate doesn't provide token counts, estimate)
            try:
                from ..utils.cost_tracker import record_cost
                # Estimate tokens based on text length (rough approximation)
                input_tokens = len(user_prompt.split()) * 1.3  # Rough token estimation
                output_tokens = len(full_response.split()) * 1.3
                record_cost(self.model, "replicate", int(input_tokens), int(output_tokens), 
                           experiment_id, dataset_name, sample_id, turn_number)
            except Exception as e:
                # Don't fail the main operation if cost tracking fails
                pass
            
            return full_response, extracted_answer, conf
            
        except Exception as e:
            error_msg = f"REPLICATE_API_ERROR: {str(e)}"
            self._safe_debug_log(q, template, error_msg, "ERROR")
            return error_msg, f"ERROR_{type(e).__name__}", 0.1

    def _safe_debug_log(self, q, tmpl, raw, parsed):
        try:
            p = Path("outputs/openai_debug.jsonl")
            p.parent.mkdir(exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                log_entry = {"q": q, "template": tmpl, "raw": raw, "parsed": parsed}
                f.write(json.dumps(log_entry) + "\n")
        except: pass
