import os, re, json
from typing import Tuple, List, Dict, Any
from pathlib import Path

def _first_number(s: str):
    m = re.search(r'[-+]?\d+(?:\.\d+)?', s or "")
    return m.group(0) if m else None

class LearnerBot:
    def __init__(self, provider: str = "demo", model: str | None = None):
        self.provider = provider
        if model:
            self.model = self._normalize_model_name(model, provider)
        elif provider == "anthropic":
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        elif provider == "openai":
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif provider == "replicate":
            self.model = os.getenv("REPLICATE_MODEL", "meta/llama-2-7b-chat")
        elif provider == "huggingface":
            self.model = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-2-7b-chat-hf")
        else:
            # Fallback for unknown providers
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    def _normalize_model_name(self, model: str, provider: str) -> str:
        """Normalize model names to their API identifiers."""
        if provider == "anthropic":
            # Map short names to full API model names
            anthropic_mapping = {
                "claude-haiku": "claude-3-haiku-20240307",
                "claude-sonnet": "claude-3-5-sonnet-20241022",  # Updated to Claude 3.5
                "claude-opus": "claude-3-opus-20240229",
                "claude-3-haiku": "claude-3-haiku-20240307",
                "claude-3-sonnet": "claude-3-5-sonnet-20241022",  # Updated to Claude 3.5
                "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",  # New mapping for 3.5
                "claude-3-opus": "claude-3-opus-20240229"
            }
            return anthropic_mapping.get(model, model)
        return model

    def answer(self, q: str, hist: List[Dict[str, Any]], template: str | None = None, 
               experiment_id: str = "unknown", dataset_name: str = "unknown", 
               sample_id: str = "unknown", turn_number: int = 0) -> Tuple[str, float, str]:
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
                demo_answer = str(int(val)) if float(val).is_integer() else f"{val:.2f}"
                demo_full_response = f"Let me solve this step by step.\n\nCalculating: {expr}\nResult: {demo_answer}\n\nTherefore, the answer is {demo_answer}."
                return demo_answer, 0.95, demo_full_response
            except:
                return "0", 0.3, "I cannot solve this problem. The answer is 0."

        if self.provider == "openai":
            return self._call_openai(q, template, experiment_id, dataset_name, sample_id, turn_number)
        elif self.provider == "anthropic":
            return self._call_anthropic(q, template, experiment_id, dataset_name, sample_id, turn_number)
        elif self.provider == "replicate":
            return self._call_replicate(q, template, experiment_id, dataset_name, sample_id, turn_number)
        elif self.provider == "huggingface":
            return self._call_huggingface(q, template, experiment_id, dataset_name, sample_id, turn_number)
        else:
            return "UNKNOWN_PROVIDER", 0.1, "Error: Unknown provider specified."

    def _call_openai(self, q: str, template: str | None = None, 
                     experiment_id: str = "unknown", dataset_name: str = "unknown", 
                     sample_id: str = "unknown", turn_number: int = 0) -> Tuple[str, float, str]:
        """Call OpenAI API."""
        from openai import OpenAI
        from ..utils.rate_limit import safe_openai_chat_completion
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Avoid global terse constraints; let dataset-level prompt dictate behavior
        sys_prompt = ""
        
        # Handle template parameter
        user_prompt = f"{q}\n[Instruction]: {template}" if template else q
        
        # Heuristic: detect code-generation tasks (HumanEval) to allow larger outputs for reasoning traces
        is_code_task = ("Python function" in q) or ("code block" in q) or ("def " in q) or ("reasoning" in q)
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2048" if is_code_task else "1024"))  # Increased for reasoning traces
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
                self._safe_debug_log(q, template, "<NULL_RESPONSE>", "ERROR")
                return "ERROR_NULL_RESPONSE", 0.1, "<NULL_RESPONSE>"
                
            text = raw_text.strip()
            if not text:
                self._safe_debug_log(q, template, "<EMPTY_RESPONSE>", "ERROR")
                return "ERROR_EMPTY_RESPONSE", 0.1, "<EMPTY_RESPONSE>"
            
            if is_code_task:
                # For reasoning traces, return full text - extraction will be done later
                ans = text  # Keep full reasoning trace
                conf = 0.6
            else:
                # For reasoning traces, return full text - extraction will be done later
                ans = text  # Keep full reasoning trace instead of just the number
                conf = 0.6
            self._safe_debug_log(q, template, text, ans, "openai")
            
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
            
            return ans, conf, text
            
        except Exception as e:
            # FIX: Don't silently fallback to "0" - log the actual error
            error_msg = f"API_ERROR: {str(e)}"
            self._safe_debug_log(q, template, error_msg, "ERROR", "openai")
            return f"ERROR_{type(e).__name__}", 0.1, error_msg

    def _call_anthropic(self, q: str, template: str | None = None, 
                        experiment_id: str = "unknown", dataset_name: str = "unknown", 
                        sample_id: str = "unknown", turn_number: int = 0) -> Tuple[str, float, str]:
        """Call Anthropic Claude API."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            
            # Handle template parameter
            user_prompt = f"{q}\n[Instruction]: {template}" if template else q
            
            # Heuristic: detect code-generation tasks
            is_code_task = ("Python function" in q) or ("code block" in q) or ("def " in q)
            max_tokens = 1024 if is_code_task else 256
            temperature = float(os.getenv("ANTHROPIC_TEMPERATURE", os.getenv("OPENAI_TEMPERATURE", "0.2")))
            
            response = client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            text = response.content[0].text.strip()
            if not text:
                return "ERROR_EMPTY_RESPONSE", 0.1, "<EMPTY_RESPONSE>"
            
            if is_code_task:
                # For reasoning traces, return full text - extraction will be done later
                ans = text  # Keep full reasoning trace
                conf = 0.6
            else:
                # For reasoning traces, return full text - extraction will be done later
                ans = text  # Keep full reasoning trace
                conf = 0.6
            
            self._safe_debug_log(q, template, text, ans, "anthropic")
            
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
            
            return ans, conf, text
            
        except Exception as e:
            error_msg = f"ANTHROPIC_API_ERROR: {str(e)}"
            self._safe_debug_log(q, template, error_msg, "ERROR", "anthropic")
            return f"ERROR_{type(e).__name__}", 0.1, error_msg

    def _call_replicate(self, q: str, template: str | None = None, 
                        experiment_id: str = "unknown", dataset_name: str = "unknown", 
                        sample_id: str = "unknown", turn_number: int = 0) -> Tuple[str, float, str]:
        """Call Replicate API (for Llama models)."""
        try:
            import replicate
            
            # Handle template parameter
            user_prompt = f"{q}\n[Instruction]: {template}" if template else q
            
            # Heuristic: detect code-generation tasks
            is_code_task = ("Python function" in q) or ("code block" in q) or ("def " in q)
            max_tokens = 1024 if is_code_task else 256
            temperature = float(os.getenv("REPLICATE_TEMPERATURE", "0.0"))
            
            response = replicate.run(
                self.model,
                input={
                    "prompt": user_prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            
            # Replicate returns a generator, so we need to collect the output
            text = ""
            for item in response:
                text += str(item)
            
            text = text.strip()
            if not text:
                return "ERROR_EMPTY_RESPONSE", 0.1, "<EMPTY_RESPONSE>"
            
            if is_code_task:
                # Extract code from markdown blocks if present
                code_match = re.search(r'```(?:python)?\n?(.*?)\n?```', text, re.DOTALL)
                if code_match:
                    ans = code_match.group(1).strip()
                else:
                    ans = text
                conf = 0.6
            else:
                # Extract numeric answer or return cleaned text
                num = _first_number(text)
                ans = num if num is not None else text[:256]
                conf = 0.85 if (num is not None and num == text.strip()) else 0.6
            
            self._safe_debug_log(q, template, text, ans)
            
            # Track cost and token usage (Replicate doesn't provide token counts, estimate)
            try:
                from ..utils.cost_tracker import record_cost
                # Estimate tokens based on text length (rough approximation)
                input_tokens = len(user_prompt.split()) * 1.3  # Rough token estimation
                output_tokens = len(text.split()) * 1.3
                record_cost(self.model, "replicate", int(input_tokens), int(output_tokens), 
                           experiment_id, dataset_name, sample_id, turn_number)
            except Exception as e:
                # Don't fail the main operation if cost tracking fails
                pass
            
            return ans, conf, text
            
        except Exception as e:
            error_msg = f"REPLICATE_API_ERROR: {str(e)}"
            self._safe_debug_log(q, template, error_msg, "ERROR")
            return f"ERROR_{type(e).__name__}", 0.1, error_msg

    def _call_huggingface(self, q: str, template: str | None = None, 
                         experiment_id: str = "unknown", dataset_name: str = "unknown", 
                         sample_id: str = "unknown", turn_number: int = 0) -> Tuple[str, float, str]:
        """Call HuggingFace API for Llama models."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Check if model is loaded in memory, if not load it
            if not hasattr(self, '_hf_model') or not hasattr(self, '_hf_tokenizer'):
                model_id = self.model or "meta-llama/Llama-2-7b-chat-hf"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self._hf_model = model
                self._hf_tokenizer = tokenizer
            
            # Prepare prompt
            if template:
                user_prompt = template.format(question=q)
            else:
                user_prompt = f"Question: {q}\nAnswer:"
            
            # Tokenize input
            inputs = self._hf_tokenizer.encode(user_prompt, return_tensors="pt")
            inputs = inputs.to(self._hf_model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self._hf_model.generate(
                    inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self._hf_tokenizer.eos_token_id
                )
            
            # Decode response
            response = self._hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer from response
            if "Answer:" in response:
                text = response.split("Answer:")[-1].strip()
            else:
                text = response[len(user_prompt):].strip()
            
            if not text:
                return "ERROR_EMPTY_RESPONSE", 0.1, "<EMPTY_RESPONSE>"
            
            # Check if this is a code task
            is_code_task = "def " in q or "function" in q.lower() or "code" in q.lower()
            
            if is_code_task:
                # Extract code from markdown blocks if present
                code_match = re.search(r'```(?:python)?\n?(.*?)\n?```', text, re.DOTALL)
                if code_match:
                    ans = code_match.group(1).strip()
                else:
                    ans = text
                conf = 0.6
            else:
                # Extract numeric answer or return cleaned text
                num = _first_number(text)
                ans = num if num is not None else text[:256]
                conf = 0.85 if (num is not None and num == text.strip()) else 0.6
            
            self._safe_debug_log(q, template, text, ans)
            
            # Track cost and token usage
            try:
                from ..utils.cost_tracker import record_cost
                input_tokens = len(inputs[0])
                output_tokens = len(outputs[0]) - len(inputs[0])
                record_cost(self.model, "huggingface", int(input_tokens), int(output_tokens), 
                           experiment_id, dataset_name, sample_id, turn_number)
            except Exception as e:
                # Don't fail the main operation if cost tracking fails
                pass
            
            return ans, conf, text
            
        except Exception as e:
            error_msg = f"HUGGINGFACE_API_ERROR: {str(e)}"
            self._safe_debug_log(q, template, error_msg, "ERROR")
            return f"ERROR_{type(e).__name__}", 0.1, error_msg

    def _safe_debug_log(self, q, tmpl, raw, parsed, provider=None):
        try:
            # Use provider-specific debug file if available, otherwise use openai
            debug_file = f"outputs/{provider}_debug.jsonl" if provider else "outputs/openai_debug.jsonl"
            p = Path(debug_file)
            p.parent.mkdir(exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                log_entry = {"q": q, "template": tmpl, "raw": raw, "parsed": parsed, "provider": provider or "openai"}
                f.write(json.dumps(log_entry) + "\n")
        except: pass
