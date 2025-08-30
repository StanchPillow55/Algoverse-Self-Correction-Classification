import os, re, json
from typing import Tuple, List, Dict, Any
from pathlib import Path

def _first_number(s: str):
    m = re.search(r'[-+]?\d+(?:\.\d+)?', s or "")
    return m.group(0) if m else None

class LearnerBot:
    def __init__(self, provider: str = "demo", model: str | None = None):
        self.provider = provider
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def answer(self, q: str, hist: List[Dict[str, Any]], template: str | None = None) -> Tuple[str, float]:
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
                return (str(int(val)) if float(val).is_integer() else f"{val:.2f}"), 0.95
            except:
                return "0", 0.3

        if self.provider == "openai":
            from openai import OpenAI
            from ..utils.rate_limit import safe_openai_chat_completion
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # Avoid global terse constraints; let dataset-level prompt dictate behavior
            sys_prompt = ""
            
            # Handle template parameter
            user_prompt = f"{q}\n[Instruction]: {template}" if template else q
            
            # Heuristic: detect code-generation tasks (HumanEval) to allow larger outputs and avoid numeric-only parsing
            is_code_task = ("Python function" in q) or ("code block" in q) or ("def " in q)
            max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1024" if is_code_task else "256"))
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
                    return "ERROR_NULL_RESPONSE", 0.1
                    
                text = raw_text.strip()
                if not text:
                    self._safe_debug_log(q, template, "<EMPTY_RESPONSE>", "ERROR")
                    return "ERROR_EMPTY_RESPONSE", 0.1
                
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
                return ans, conf
                
            except Exception as e:
                # FIX: Don't silently fallback to "0" - log the actual error
                error_msg = f"API_ERROR: {str(e)}"
                self._safe_debug_log(q, template, error_msg, "ERROR")
                return f"ERROR_{type(e).__name__}", 0.1

        return "UNKNOWN_PROVIDER", 0.1  # Don't default to "0"

    def _safe_debug_log(self, q, tmpl, raw, parsed):
        try:
            p = Path("outputs/openai_debug.jsonl")
            p.parent.mkdir(exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                log_entry = {"q": q, "template": tmpl, "raw": raw, "parsed": parsed}
                f.write(json.dumps(log_entry) + "\n")
        except: pass
