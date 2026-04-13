from __future__ import annotations

from typing import Optional


class HotfixService:
    def __init__(self, llm_client, log_fn):
        self._llm_client = llm_client
        self._log = log_fn

    def generate_fix(self, original_code: str, error_message: str) -> Optional[str]:
        if self._llm_client is None:
            return None
        try:
            hotfix_prompt = f"""You are a Python code debugging expert. Fix the following code that has an error.

ORIGINAL CODE:
```python
{original_code}
```

ERROR MESSAGE:
{error_message}

Please provide ONLY the corrected Python code without any explanations or markdown formatting.
Return only the corrected Python code:"""
            response = self._llm_client.responses.create(
                model="gpt-5-mini",
                instructions=(
                    "You are a Python debugging expert. Fix the code and return only corrected Python code."
                ),
                input=hotfix_prompt,
            )
            fixed_code = response.output_text.strip()
            if fixed_code.startswith("```python"):
                fixed_code = fixed_code.replace("```python", "").replace("```", "").strip()
            elif fixed_code.startswith("```"):
                fixed_code = fixed_code.replace("```", "").strip()
            return fixed_code or None
        except Exception as exc:
            self._log(f"Hotfix generation failed: {exc}")
            return None
