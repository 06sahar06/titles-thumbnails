Clickbait detection project

Setup

1. Activate your Python venv (you already have one at `thumbnails & titles/.venv`).

   Windows PowerShell:

   ```powershell
   (Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& "thumbnails & titles\.venv\Scripts\Activate.ps1")
   ```

2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

Quick checks

- Run the smoke checks (verifies imports, CSVs, thumbnails, CUDA availability):

```powershell
python run_smoke_checks.py
```

Notes on API keys

- `neutralize_titles.py` and `openrouter_zero_shot.py` call the OpenRouter API and require `--api-key` (or pass `--base-url` for a specific endpoint). Provide your OpenRouter API key when invoking those scripts.
- `vllm_zero_shot.py` expects a vLLM/OpenRouter-like endpoint; if you use a hosted provider, pass `--api-key` and `--base-url` accordingly.

Files added by this setup

- `requirements.txt` — Python dependencies
- `run_smoke_checks.py` — quick environment verification script

Next steps

- I will install dependencies and run the smoke checks unless you want to change anything first. If you do, tell me which package versions or which scripts to skip.