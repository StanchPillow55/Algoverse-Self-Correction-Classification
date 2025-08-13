
import os, re, json
from typing import Dict, Any, Tuple, List
from pathlib import Path

def _first_number(s: str) -> str | None:
    """Safely extracts the first floating-point number from a string."""
    m = re.search(r'[-+]?\d+(?:\.\d+)?', s)
    return m.group(0) if m else None

class LearnerBot:
    def __init__(self, provider: str = "demo", temperature: float = 0.3, model: str | None = None):
        self.provider = provider
        self.temperature = temperature
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def answer(self, question: str, history: List[Dict[str, Any]], template: str | None = None) -> Tuple[str, float]:
        # DEMO PATH: Local arithmetic for quick tests
        if (os.getenv("DEMO_MODE", "1") == "1") or self.provider == "demo":
            expr = re.sub(r'[^0-9+\-*/. ]', '', question)
            if re.search(r'[0-9]', expr) and re.search(r'[+\-*/]', expr):
                try:
                    val = eval(expr)
                    out = str(int(val)) if float(val).is_integer() else f"{val:.4f}".rstrip('0').rstrip('.')
                    return out, 0.95
                except Exception:
                    pass
            return "0", 0.3

        # OPENAI PATH: Real API call for number-only replies
        if self.provider == "openai":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                sys_prompt = "You are a precise calculator. Return ONLY the final numeric answer. No words, no explanations, no currency symbols. Just the number."
                user_prompt = question
                if template:
                    user_prompt = f"{user_prompt}\n\n[Instruction]: {template}"

                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": sys_prompt},
                              {"role": "user", "content": user_prompt}],
                    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
                    max_tokens=20
                )
                text = (resp.choices[0].message.content or "").strip()
                num = _first_number(text)
                ans = num if num is not None else text[:24] # Fallback to raw text if no number found
                conf = 0.85 if num is not None and num == text else 0.6 # Confidence is higher for clean numeric replies

                # Safe debug logging for the first 3 interactions
                dbg_path = Path("outputs/openai_debug.json")
                blob = json.loads(dbg_path.read_text()) if dbg_path.exists() else []
                if len(blob) < 3:
                    blob.append({"question": question, "template": template, "raw_response": text, "parsed_answer": ans})
                    dbg_path.write_text(json.dumps(blob, indent=2))
                return ans, conf
            except Exception as e:
                print(f"!! OpenAI call failed: {e}", flush=True)
                return "0", 0.1 # Return low-confidence zero on error

        # Fallback for any unknown provider
        return "0", 0.3
