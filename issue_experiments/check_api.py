"""
Minimal API connectivity check -- standard OpenAI SDK format.

Reads the key (and optional base_url / model) from .env via llmcer.config and
makes ONE tiny chat completion call. Prints exactly what comes back, including
the precise error if it fails, so we can tell whether the key/endpoint works.

Run:  .venv/Scripts/python.exe issue_experiments/check_api.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmcer.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from openai import OpenAI


def main():
    k = OPENAI_API_KEY or ""
    print("=== Config ===")
    print(f"key length   : {len(k)}")
    print(f"key prefix   : {k[:7]!r}")
    print(f"key suffix   : {k[-4:]!r}")
    print(f"whitespace?  : {any(c.isspace() for c in k)}")
    print(f"base_url     : {OPENAI_BASE_URL!r}  (None = official api.openai.com)")
    print(f"model        : {OPENAI_MODEL!r}")
    print()

    if not k or k == "your_api_key_here":
        print("ERROR: no API key in .env (OPENAI_API_KEY=...). Aborting.")
        return 2

    if OPENAI_BASE_URL:
        client = OpenAI(api_key=k, base_url=OPENAI_BASE_URL)
    else:
        client = OpenAI(api_key=k)

    print("=== Calling chat.completions.create (1 tiny request) ===")
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=5,
            temperature=0,
        )
        print("SUCCESS")
        print("  content :", repr(resp.choices[0].message.content))
        print("  model   :", resp.model)
        print("  usage   :", resp.usage)
        return 0
    except Exception as e:
        print("FAILED")
        print(f"  type   : {type(e).__name__}")
        print(f"  detail : {e}")
        status = getattr(e, "status_code", None)
        if status is not None:
            print(f"  status : {status}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
