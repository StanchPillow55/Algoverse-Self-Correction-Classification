import os, re, json
from typing import Tuple, List, Dict, Any
from pathlib import Path
from collections import Counter
import random
import time

def _first_number(s: str):
    m = re.search(r'[-+]?\d+(?:\.\d+)?', s or "")
    return m.group(0) if m else None

class EnsembleLearnerBot:
    def __init__(self, provider: str = "demo", model: str | None = None, ensemble_size: int = 3, 
                 ensemble_models: List[str] = None, ensemble_configs: List[Dict[str, str]] = None,
                 error_handler = None):
        self.provider = provider
        self.ensemble_size = ensemble_size
        
        # Support for heterogeneous ensembles with different providers
        if ensemble_configs:
            # Heterogeneous ensemble: each model can have different provider
            self.ensemble_configs = ensemble_configs[:ensemble_size]
            self.ensemble_models = [config.get('model', 'unknown') for config in self.ensemble_configs]
            self.is_heterogeneous = True
        elif ensemble_models:
            # Homogeneous ensemble: all models use same provider
            self.ensemble_models = ensemble_models[:ensemble_size]
            self.ensemble_configs = [{'provider': provider, 'model': model} for model in self.ensemble_models]
            self.is_heterogeneous = False
        else:
            # Default ensemble configurations per provider
            if provider == "anthropic":
                default_models = [
                    "claude-3-haiku-20240307", 
                    "claude-3-5-sonnet-20241210", 
                    "claude-3-opus-20240229"
                ]
            elif provider == "openai":
                default_models = [
                    "gpt-4o-mini", 
                    "gpt-4o", 
                    "gpt-3.5-turbo"
                ]
            else:
                # For demo or other providers, use repetition with variation
                base_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                default_models = [base_model] * ensemble_size
            
            self.ensemble_models = default_models[:ensemble_size]
            self.ensemble_configs = [{'provider': provider, 'model': model} for model in self.ensemble_models]
            self.is_heterogeneous = False
        
        # Normalize model names for each provider
        for i, config in enumerate(self.ensemble_configs):
            config['model'] = self._normalize_model_name(config['model'], config['provider'])
            self.ensemble_models[i] = config['model']
        
        # Keep primary model for compatibility
        self.model = self.ensemble_models[0] if self.ensemble_models else "gpt-4o-mini"
        
        # Error handler for API resilience
        self.error_handler = error_handler
        
        # Track ensemble health
        self.ensemble_errors = {f"{config['provider']}:{config['model']}": 0 for config in self.ensemble_configs}
    
    def _normalize_model_name(self, model: str, provider: str) -> str:
        """Normalize model names to their API identifiers."""
        if provider == "anthropic":
            # Map short names to full API model names
            anthropic_mapping = {
                "claude-haiku": "claude-3-haiku-20240307",
                "claude-sonnet": "claude-3-5-sonnet-20241210",  # Updated to Claude 3.5
                "claude-opus": "claude-3-opus-20240229",
                "claude-3-haiku": "claude-3-haiku-20240307",
                "claude-3-sonnet": "claude-3-5-sonnet-20241210",  # Updated to Claude 3.5
                "claude-3.5-sonnet": "claude-3-5-sonnet-20241210",  # New mapping for 3.5
                "claude-3-opus": "claude-3-opus-20240229"
            }
            return anthropic_mapping.get(model, model)
        return model

    def answer(self, q: str, hist: List[Dict[str, Any]], template: str | None = None, 
               experiment_id: str = "unknown", dataset_name: str = "unknown", 
               sample_id: str = "unknown", turn_number: int = 0) -> Tuple[str, float, str]:
        """Ensemble answer method using majority voting across multiple models"""
        
        # Special handling for demo mode - use single model behavior
        if os.getenv("DEMO_MODE", "0") == "1" or self.provider == "demo":
            return self._demo_answer(q)
        
        # Collect responses from all ensemble models
        ensemble_responses = []
        ensemble_confidences = []
        ensemble_raw_texts = []
        
        for i, config in enumerate(self.ensemble_configs):
            model = config['model']
            model_provider = config['provider']
            provider_key = f"{model_provider}:{model}"
            
            # Skip this model if error handler says we should
            if self.error_handler and self.error_handler.should_skip_sample(f"{sample_id}_ensemble_{i}"):
                print(f"Skipping {provider_key} due to previous errors")
                ensemble_responses.append(f"SKIPPED_TOO_MANY_ERRORS")
                ensemble_confidences.append(0.1)
                ensemble_raw_texts.append(f"Model {i+1} ({provider_key}): SKIPPED - too many previous errors")
                continue
            
            try:
                # Get response from this specific model/provider combination
                answer, confidence, raw_text = self._call_heterogeneous_model_resilient(
                    q, template, model_provider, model, experiment_id, dataset_name, 
                    f"{sample_id}_ensemble_{i}", turn_number
                )
                
                ensemble_responses.append(answer)
                ensemble_confidences.append(confidence)
                ensemble_raw_texts.append(f"Model {i+1} ({model_provider}:{model}): {raw_text}")
                
            except Exception as e:
                # Handle the error with error handler if available
                if self.error_handler:
                    should_continue, backoff_delay = self.error_handler.handle_api_error(
                        e, model_provider, model, experiment_id, f"{sample_id}_ensemble_{i}", turn_number
                    )
                    
                    if not should_continue:
                        # Error handler decided to terminate
                        raise e
                    
                    if backoff_delay and backoff_delay > 0:
                        print(f"Backing off for {backoff_delay:.1f}s after error from {provider_key}")
                        time.sleep(backoff_delay)
                
                # Track ensemble errors
                self.ensemble_errors[provider_key] = self.ensemble_errors.get(provider_key, 0) + 1
                
                print(f"Warning: Ensemble model {model_provider}:{model} failed: {e}")
                ensemble_responses.append(f"ERROR_{type(e).__name__}")
                ensemble_confidences.append(0.1)
                ensemble_raw_texts.append(f"Model {i+1} ({model_provider}:{model}): ERROR - {str(e)}")
        
        # Check ensemble health if error handler is available
        if self.error_handler and not self.error_handler.check_ensemble_health(self.ensemble_errors):
            print(f"🚨 Warning: Ensemble health degraded - too many model failures")
            print(f"Error counts: {self.ensemble_errors}")
        
        # Determine voting strategy from configuration or default to adaptive
        voting_strategy = os.getenv('ENSEMBLE_VOTING_STRATEGY', 'adaptive')
        is_code_task = ("Python function" in q) or ("code block" in q) or ("def " in q)
        
        # Perform ensemble aggregation using selected strategy
        if voting_strategy == 'majority_with_confidence':
            final_answer, final_confidence, aggregation_info = self._aggregate_ensemble_responses(
                ensemble_responses, ensemble_confidences, ensemble_raw_texts
            )
        elif voting_strategy == 'weighted_confidence':
            final_answer, final_confidence, aggregation_info = self._weighted_confidence_voting(
                ensemble_responses, ensemble_confidences
            )
        elif voting_strategy == 'consensus_detection':
            final_answer, final_confidence, aggregation_info = self._consensus_detection_voting(
                ensemble_responses, ensemble_confidences
            )
        elif voting_strategy == 'adaptive':
            final_answer, final_confidence, aggregation_info = self._adaptive_voting(
                ensemble_responses, ensemble_confidences, ensemble_raw_texts, is_code_task
            )
        else:
            # Default to majority with confidence
            final_answer, final_confidence, aggregation_info = self._aggregate_ensemble_responses(
                ensemble_responses, ensemble_confidences, ensemble_raw_texts
            )
        
        # Create comprehensive response text
        combined_response = self._format_ensemble_response(
            ensemble_raw_texts, final_answer, final_confidence, aggregation_info
        )
        
        return final_answer, final_confidence, combined_response
    
    def _demo_answer(self, q: str) -> Tuple[str, float, str]:
        """Demo mode answer - simplified for testing"""
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
    
    def _call_heterogeneous_model_resilient(self, q: str, template: str | None = None, 
                                           provider: str = None, model: str = None,
                                           experiment_id: str = "unknown", dataset_name: str = "unknown", 
                                           sample_id: str = "unknown", turn_number: int = 0) -> Tuple[str, float, str]:
        """Call a specific model from a specific provider with error handling - supports heterogeneous ensembles"""
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                return self._call_heterogeneous_model(
                    q, template, provider, model, experiment_id, dataset_name, sample_id, turn_number
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    # Final attempt - let the error propagate
                    raise e
                
                # Check if this is a retriable error
                error_str = str(e).lower()
                if any(non_retriable in error_str for non_retriable in ['authentication', 'invalid api key', 'quota']):
                    # Don't retry non-retriable errors
                    raise e
                
                # Wait before retry with exponential backoff
                delay = base_delay * (2 ** attempt)
                print(f"Attempt {attempt + 1} failed for {provider}:{model}, retrying in {delay}s: {e}")
                time.sleep(delay)
        
        # Should never reach here, but just in case
        raise Exception("Max retries exceeded")
    
    def _call_heterogeneous_model(self, q: str, template: str | None = None, 
                                 provider: str = None, model: str = None,
                                 experiment_id: str = "unknown", dataset_name: str = "unknown", 
                                 sample_id: str = "unknown", turn_number: int = 0) -> Tuple[str, float, str]:
        """Call a specific model from a specific provider - supports heterogeneous ensembles"""
        
        if os.getenv("DEMO_MODE", "0") == "1" or provider == "demo":
            return self._demo_answer(q)
        
        # Temporarily set the model and provider for this call
        original_model = self.model
        original_provider = self.provider
        
        try:
            self.model = model
            self.provider = provider
            
            if provider == "openai":
                return self._call_openai(q, template, experiment_id, dataset_name, sample_id, turn_number)
            elif provider == "anthropic":
                return self._call_anthropic(q, template, experiment_id, dataset_name, sample_id, turn_number)
            elif provider == "replicate":
                return self._call_replicate(q, template, experiment_id, dataset_name, sample_id, turn_number)
            elif provider == "huggingface":
                return self._call_huggingface(q, template, experiment_id, dataset_name, sample_id, turn_number)
            else:
                return f"UNKNOWN_PROVIDER_{provider}", 0.1, f"Error: Unknown provider {provider} specified."
        
        finally:
            # Always restore original model and provider
            self.model = original_model
            self.provider = original_provider
    
    def _call_single_model(self, q: str, template: str | None = None, 
                          experiment_id: str = "unknown", dataset_name: str = "unknown", 
                          sample_id: str = "unknown", turn_number: int = 0) -> Tuple[str, float, str]:
        """Call a single model - routes to appropriate provider method (legacy method for homogeneous ensembles)"""
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
    
    def _aggregate_ensemble_responses(self, responses: List[str], confidences: List[float], 
                                    raw_texts: List[str]) -> Tuple[str, float, Dict[str, Any]]:
        """Aggregate ensemble responses using majority voting and confidence weighting"""
        
        # Filter out error responses
        valid_responses = []
        valid_confidences = []
        valid_indices = []
        
        for i, (resp, conf) in enumerate(zip(responses, confidences)):
            if not resp.startswith("ERROR_"):
                valid_responses.append(resp)
                valid_confidences.append(conf)
                valid_indices.append(i)
        
        if not valid_responses:
            # All responses failed - return error
            return "ERROR_ALL_MODELS_FAILED", 0.1, {
                "voting_method": "error_fallback",
                "valid_responses": 0,
                "total_responses": len(responses),
                "error_details": responses
            }
        
        # Perform majority voting
        response_counts = Counter(valid_responses)
        most_common_response, vote_count = response_counts.most_common(1)[0]
        
        # Calculate confidence based on consensus and individual confidences
        consensus_ratio = vote_count / len(valid_responses)
        avg_confidence = sum(valid_confidences) / len(valid_confidences)
        
        # Weighted confidence: consensus strength + average model confidence
        final_confidence = (consensus_ratio * 0.6) + (avg_confidence * 0.4)
        final_confidence = min(final_confidence, 0.95)  # Cap confidence
        
        # Handle ties with confidence weighting
        if len(response_counts) > 1 and list(response_counts.values())[0] == list(response_counts.values())[1]:
            # Tie detected - use confidence-weighted voting
            response_conf_sums = {}
            for i, resp in enumerate(valid_responses):
                if resp not in response_conf_sums:
                    response_conf_sums[resp] = 0
                response_conf_sums[resp] += valid_confidences[i]
            
            # Choose response with highest confidence sum
            most_common_response = max(response_conf_sums.keys(), key=response_conf_sums.get)
            final_confidence *= 0.9  # Reduce confidence due to tie
        
        aggregation_info = {
            "voting_method": "majority_with_confidence_tiebreak" if len(response_counts) > 1 else "majority",
            "valid_responses": len(valid_responses),
            "total_responses": len(responses),
            "consensus_ratio": round(consensus_ratio, 3),
            "response_distribution": dict(response_counts),
            "confidence_range": [min(valid_confidences), max(valid_confidences)],
            "avg_individual_confidence": round(avg_confidence, 3),
            "failed_models": len(responses) - len(valid_responses)
        }
        
        return most_common_response, final_confidence, aggregation_info
    
    def _format_ensemble_response(self, raw_texts: List[str], final_answer: str, 
                                final_confidence: float, aggregation_info: Dict[str, Any]) -> str:
        """Format comprehensive ensemble response showing all model outputs and voting results"""
        
        response_parts = [
            "=== ENSEMBLE RESPONSE ===",
            f"Final Answer: {final_answer}",
            f"Ensemble Confidence: {final_confidence:.3f}",
            f"Voting Method: {aggregation_info.get('voting_method', aggregation_info.get('method', 'majority'))}",
            f"Consensus: {aggregation_info.get('valid_responses', 0)}/{aggregation_info.get('total_responses', len(raw_texts))} models agreed",
            "",
            "=== INDIVIDUAL MODEL RESPONSES ==="
        ]
        
        for i, raw_text in enumerate(raw_texts):
            response_parts.append(f"\n--- Model {i+1} Response ---")
            response_parts.append(raw_text[:500] + "..." if len(raw_text) > 500 else raw_text)
        
        if aggregation_info.get('response_distribution'):
            response_parts.extend([
                "",
                "=== VOTING SUMMARY ===",
                f"Response Distribution: {aggregation_info['response_distribution']}",
                f"Consensus Ratio: {aggregation_info['consensus_ratio']}",
                f"Average Individual Confidence: {aggregation_info['avg_individual_confidence']}"
            ])
        
        if aggregation_info.get('failed_models', 0) > 0:
            response_parts.append(f"Failed Models: {aggregation_info['failed_models']}")
        
        return "\n".join(response_parts)
    
    def _weighted_confidence_voting(self, responses: List[str], confidences: List[float]) -> Tuple[str, float, Dict[str, Any]]:
        """Weighted voting based on model confidence scores"""
        response_conf_map = {}
        
        for resp, conf in zip(responses, confidences):
            if not resp.startswith("ERROR_"):
                if resp not in response_conf_map:
                    response_conf_map[resp] = []
                response_conf_map[resp].append(conf)
        
        if not response_conf_map:
            return "ERROR_ALL_MODELS_FAILED", 0.1, {"method": "weighted_confidence", "error": "no_valid_responses"}
        
        # Calculate weighted scores for each unique response
        weighted_scores = {}
        for resp, confs in response_conf_map.items():
            weighted_scores[resp] = sum(confs) / len(confs) * len(confs)  # avg_conf * count
        
        best_response = max(weighted_scores.keys(), key=weighted_scores.get)
        final_confidence = min(weighted_scores[best_response] / len(responses), 0.95)
        
        return best_response, final_confidence, {
            "method": "weighted_confidence",
            "response_scores": weighted_scores,
            "winner": best_response,
            "winner_score": weighted_scores[best_response]
        }
    
    def _consensus_detection_voting(self, responses: List[str], confidences: List[float], 
                                   similarity_threshold: float = 0.8) -> Tuple[str, float, Dict[str, Any]]:
        """Detect consensus using text similarity for code/text responses"""
        import difflib
        
        valid_responses = [(resp, conf) for resp, conf in zip(responses, confidences) 
                          if not resp.startswith("ERROR_")]
        
        if not valid_responses:
            return "ERROR_ALL_MODELS_FAILED", 0.1, {"method": "consensus_detection", "error": "no_valid_responses"}
        
        if len(valid_responses) == 1:
            return valid_responses[0][0], valid_responses[0][1], {
                "method": "consensus_detection",
                "consensus_type": "single_response",
                "similarity_scores": []
            }
        
        # Calculate pairwise similarities
        similarity_matrix = []
        for i, (resp_i, _) in enumerate(valid_responses):
            similarities = []
            for j, (resp_j, _) in enumerate(valid_responses):
                if i == j:
                    similarities.append(1.0)
                else:
                    sim = difflib.SequenceMatcher(None, resp_i.lower(), resp_j.lower()).ratio()
                    similarities.append(sim)
            similarity_matrix.append(similarities)
        
        # Find responses with high consensus
        consensus_groups = []
        used_indices = set()
        
        for i in range(len(valid_responses)):
            if i in used_indices:
                continue
            
            group = [i]
            group_confidences = [valid_responses[i][1]]
            
            for j in range(i + 1, len(valid_responses)):
                if j in used_indices:
                    continue
                
                if similarity_matrix[i][j] >= similarity_threshold:
                    group.append(j)
                    group_confidences.append(valid_responses[j][1])
                    used_indices.add(j)
            
            used_indices.add(i)
            consensus_groups.append((group, group_confidences))
        
        # Select the largest consensus group
        best_group = max(consensus_groups, key=lambda x: len(x[0]))
        best_indices, best_confidences = best_group
        
        # Choose representative response (highest confidence in the group)
        best_idx_in_group = max(range(len(best_indices)), key=lambda x: best_confidences[x])
        final_response = valid_responses[best_indices[best_idx_in_group]][0]
        
        # Calculate consensus strength
        consensus_strength = len(best_indices) / len(valid_responses)
        avg_group_confidence = sum(best_confidences) / len(best_confidences)
        final_confidence = min((consensus_strength * 0.7) + (avg_group_confidence * 0.3), 0.95)
        
        return final_response, final_confidence, {
            "method": "consensus_detection",
            "consensus_groups": len(consensus_groups),
            "largest_group_size": len(best_indices),
            "consensus_strength": round(consensus_strength, 3),
            "similarity_threshold": similarity_threshold,
            "avg_similarity_in_group": round(sum(similarity_matrix[best_indices[0]][j] 
                                               for j in best_indices) / len(best_indices), 3)
        }
    
    def _adaptive_voting(self, responses: List[str], confidences: List[float], 
                        raw_texts: List[str], is_code_task: bool = False) -> Tuple[str, float, Dict[str, Any]]:
        """Adaptive voting that chooses strategy based on response characteristics"""
        
        # Filter valid responses
        valid_count = sum(1 for resp in responses if not resp.startswith("ERROR_"))
        
        if valid_count == 0:
            return "ERROR_ALL_MODELS_FAILED", 0.1, {"method": "adaptive", "chosen_strategy": "error_fallback"}
        
        # Calculate response diversity
        unique_responses = len(set(resp for resp in responses if not resp.startswith("ERROR_")))
        diversity_ratio = unique_responses / max(valid_count, 1)
        
        # Calculate confidence spread
        valid_confidences = [conf for resp, conf in zip(responses, confidences) if not resp.startswith("ERROR_")]
        conf_std = (max(valid_confidences) - min(valid_confidences)) if len(valid_confidences) > 1 else 0
        
        # Choose strategy based on characteristics
        if diversity_ratio <= 0.5:  # Low diversity - use majority voting
            chosen_strategy = "majority_with_confidence"
            result = self._aggregate_ensemble_responses(responses, confidences, raw_texts)
        elif conf_std > 0.3:  # High confidence spread - use confidence weighting
            chosen_strategy = "weighted_confidence"
            result = self._weighted_confidence_voting(responses, confidences)
        elif is_code_task or any(len(resp) > 200 for resp in responses if not resp.startswith("ERROR_")):  # Long responses - use consensus
            chosen_strategy = "consensus_detection"
            result = self._consensus_detection_voting(responses, confidences)
        else:  # Default to majority with confidence
            chosen_strategy = "majority_with_confidence"
            result = self._aggregate_ensemble_responses(responses, confidences, raw_texts)
        
        # Add adaptive metadata
        final_answer, final_confidence, base_info = result
        base_info.update({
            "adaptive_strategy": chosen_strategy,
            "diversity_ratio": round(diversity_ratio, 3),
            "confidence_std": round(conf_std, 3),
            "valid_responses": valid_count,
            "is_code_task": is_code_task
        })
        
        return final_answer, final_confidence, base_info
