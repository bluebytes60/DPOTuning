"""Test whether OPENAI_API_KEY is set and working."""
import os
import sys

key = os.environ.get("OPENAI_API_KEY", "")
if not key:
    print("OPENAI_API_KEY is NOT set in this environment.")
    sys.exit(1)

print(f"Key found: {key[:8]}...{key[-4:]}  (length {len(key)})")

try:
    from openai import OpenAI
except ImportError:
    print("openai package not installed. Run: pip install openai")
    sys.exit(1)

print("Making test API call to gpt-4o-mini...")
try:
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Reply with the single word: working"}],
        max_tokens=5,
    )
    print(f"Response: {resp.choices[0].message.content.strip()}")
    print("API key is valid ✓")
except Exception as e:
    print(f"API call failed: {e}")
