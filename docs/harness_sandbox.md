# Harness Sandbox

This repository executes user code (HumanEval) in a sandboxed subprocess with the following constraints:

- Timeout: 10s per candidate by default
- Isolation: Temporary working directory with a limited PYTHONPATH
- Import restrictions: denylisted potentially unsafe modules (os, sys, subprocess, socket, requests, etc.)
- Error tagging: stdout tags (FAIL_SYNTAX, FAIL_ASSERTION, FAIL_RUNTIME) to classify errors deterministically

Operational notes:
- DEMO_MODE=1 uses a mock executor with simulated results, suitable for smoke tests and offline validation
- The executor attempts to avoid dangerous builtins and file operations; see src/eval/code_executor.py
- For production runs, consider OS-level isolation (containers) for additional protection

