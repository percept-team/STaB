import hashlib
import json
import os
import re
import time
from pathlib import Path
import requests

DIMENSIONS = ["Subject Correspondence", "Appearance Description", "Temporal Description", "Hallucination Detection"]


def parse_scores(text):
    scores = {}
    for dimension in DIMENSIONS:
        match = re.search(re.escape(dimension) + r"\s*:\s*([0-5](?:\.\d+)?)", text, re.IGNORECASE)
        if not match or not 0 <= float(match.group(1)) <= 5:
            raise ValueError(f"missing or invalid judge score: {dimension}")
        scores[dimension] = float(match.group(1))
    return scores


def judge_videorefer(prediction_file, system_prompt, output, post=None):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("set OPENAI_API_KEY to run the optional paid VideoRefer judge")
    model = os.getenv("RFVL_JUDGE_MODEL", "gpt-4o")
    endpoint = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    prompt = Path(system_prompt).read_text(encoding="utf-8")
    inputs = json.loads(Path(prediction_file).read_text(encoding="utf-8"))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = json.loads(output.read_text(encoding="utf-8")) if output.exists() else []
    seen = {row["id"] for row in completed}
    post = post or requests.post
    for row in inputs:
        if row["id"] in seen:
            continue
        payload = {"model": model, "messages": [{"role": "system", "content": prompt},
                    {"role": "user", "content": "##Correct answer: " + row["caption"] + "\n##Predicted answer: " + row["pred"]}],
                   "temperature": .7, "top_p": .95, "max_tokens": 800}
        for attempt in range(5):
            response = post(endpoint, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=120)
            if response.status_code in {429, 500, 502, 503} and attempt < 4:
                time.sleep(min(2**attempt, 8))
                continue
            response.raise_for_status()
            body = response.json()
            text = body["choices"][0]["message"]["content"]
            scores = parse_scores(text)
            completed.append({**row, "scores": scores, "judge_text": text, "model": body.get("model", model),
                              "system_fingerprint": body.get("system_fingerprint"), "usage": body.get("usage"),
                              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
            temporary = output.with_suffix(".tmp")
            temporary.write_text(json.dumps(completed, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(output)
            break
    return aggregate_scores(output)


def aggregate_scores(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("no judged predictions")
    result = {dimension: sum(row["scores"][dimension] for row in rows) / len(rows) for dimension in DIMENSIONS}
    result["Avg"] = sum(result.values()) / 4
    result["samples"] = len(rows)
    return result
