#!/usr/bin/env python3
"""Minimal client showing how CosMet talks to the local server. Requires `openai`.
Timing fields reproduce the two figures in the paper (see hardware.md)."""
import time, sys
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

def ask(system_prompt: str, user_content: str):
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_content}],
        temperature=0.0, top_p=1.0, presence_penalty=0.0, frequency_penalty=0.0, max_tokens=8192,
        extra_body={"reasoning_effort": "medium"},
    )
    dt = time.perf_counter() - t0
    assert r.choices[0].finish_reason != "length", "output truncated: raise max_tokens"
    out_tok = r.usage.completion_tokens          # includes reasoning tokens in vLLM
    return r.choices[0].message.content, dt, out_tok / dt   # text, wall-clock s, tok/s

if __name__ == "__main__":
    text, secs, tps = ask(open("../prompts/analyzer_skeleton.md").read(), sys.stdin.read())
    print(text); print(f"\n[{secs:.1f} s end-to-end, {tps:.0f} tok/s decode]", file=sys.stderr)
