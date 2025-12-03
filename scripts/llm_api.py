#!/usr/bin/env python3
"""
Ball Knower v2 — LLM API Execution Layer

This script makes deterministic, logged API calls to LLMs.
All context is explicit. Nothing is accumulated.

Usage:
    python scripts/llm_api.py --role planner --task "BK-TASK-001" --context context.txt
    python scripts/llm_api.py --role auditor --task "BK-TASK-001" --context context.txt
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def call_claude(prompt: str, task_id: str) -> dict:
    """Call Claude API for planning tasks."""
    from anthropic import Anthropic

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system_prompt = """You are the Planning Agent for Ball Knower v2.

Rules:
- You have no memory of previous conversations
- Every claim must be cited with file:line or command->output
- Forbidden phrases without citation: "should be", "typically", "I believe", "probably"
- If you cannot cite a claim, state "UNVERIFIED: [claim]"
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "role": "planner",
        "model": "claude-sonnet-4-20250514",
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": response.content[0].text,
    }


def call_gpt(prompt: str, task_id: str) -> dict:
    """Call GPT API for audit tasks."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = """You are the Outside Auditor for Ball Knower v2.

Rules:
- You have no memory of previous conversations
- You must verify all claims against evidence provided
- Challenge any assumption that lacks citation
- Your responses: APPROVED, APPROVED WITH NOTES, or BLOCKED
- If blocked, state exactly what evidence is missing
"""

    response = client.chat.completions.create(
        model="gpt-4o",  # Update to latest GPT model as needed
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    return {
        "role": "auditor",
        "model": "gpt",
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": response.choices[0].message.content,
    }


def main():
    parser = argparse.ArgumentParser(description="BK v2 LLM API Executor")
    parser.add_argument("--role", choices=["planner", "auditor"], required=True)
    parser.add_argument("--task", required=True, help="Task ID (e.g., BK-TASK-001)")
    parser.add_argument("--context", required=True, help="Path to context file")
    parser.add_argument("--prompt", help="Additional prompt text")
    args = parser.parse_args()

    # Read context file
    context_path = Path(args.context)
    if not context_path.exists():
        print(f"Error: Context file not found: {args.context}")
        return 1

    context = context_path.read_text()

    # Build prompt
    prompt = f"Task: {args.task}\n\n"
    prompt += f"Context:\n{context}\n\n"
    if args.prompt:
        prompt += f"Instructions:\n{args.prompt}\n"

    # Make API call
    if args.role == "planner":
        result = call_claude(prompt, args.task)
    else:
        result = call_gpt(prompt, args.task)

    # Log the call
    log_dir = Path("data/llm_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (
        log_dir
        / f"{args.task}_{args.role}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(log_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Role: {result['role'].upper()}")
    print(f"Task: {result['task_id']}")
    print(f"Model: {result['model']}")
    print(f"Log: {log_file}")
    print(f"{'='*60}\n")
    print(result["response"])
    print(f"\n{'='*60}")

    return 0


if __name__ == "__main__":
    exit(main())
