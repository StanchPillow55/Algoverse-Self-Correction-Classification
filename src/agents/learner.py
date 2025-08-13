import os, re
from typing import Dict, Any, Tuple, List

class LearnerBot:
    def __init__(self, provider: str = "demo", temperature: float = 0.3):
        self.provider = provider
        self.temperature = temperature

    def answer(self, question: str, history: List[Dict[str, Any]], template: str | None = None) -> Tuple[str, float]:
        """
        Returns (answer_text, self_conf) with self_conf in [0,1].
        Demo provider: handles simple arithmetic robustly; else echoes a safe guess.
        """
        if (os.getenv("DEMO_MODE", "1") == "1") or self.provider == "demo":
            # Try to extract a simple arithmetic expression
            expr = self._extract_expr(question)
            if expr is not None:
                try:
                    val = eval(expr)
                    return (str(int(val)) if float(val).is_integer() else str(val)), 0.9
                except Exception:
                    pass
            # fallback: guess '0' with low confidence
            return "0", 0.3
        else:
            # Placeholder for real providers (OpenAI/Anthropic) to be added later
            return "0", 0.3

    def _extract_expr(self, text: str) -> str | None:
        # crude parse: keep digits and +-*/ plus spaces
        s = re.sub(r"[^0-9+\-*/. ]", "", text)
        # basic sanity: must contain a digit and an operator
        return s if re.search(r"[0-9]", s) and re.search(r"[+\-*/]", s) else None
