import itertools
import json
import os
import re
import time
import zlib
from pathlib import Path

import requests
from omegaconf import OmegaConf
from tqdm import tqdm

API = "https://www.neuronpedia.org/api"
PROMPTS = Path(__file__).parent.parent / "prompts"
TASKS = {task: json.loads((PROMPTS / f"{task}.json").read_text()) for task in ["2way", "3way"]}
GASLIGHT = (PROMPTS / "gaslight.txt").read_text()


def result_dir(cfg, kind):
    root = Path(cfg.results_dir) / kind / cfg[kind].experiment_name
    path = root / cfg.timestamp
    path.mkdir(parents=True, exist_ok=True)
    latest = root / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(cfg.timestamp)
    OmegaConf.save(cfg, path / "config.yaml", resolve=True)
    return path


def make_prompt(task, order, concept=None):
    # Option texts fill the template in the given order, as utils.make_prompt_string in the official repo.
    d = TASKS[task]
    prompt = d["head_text"].format(*[d[k][i] for i in range(len(d[order[0]])) for k in order])
    if concept is not None:
        prompt = GASLIGHT.replace("{}", concept) + prompt
    return prompt


def parse(response, order, mode):
    # Same rules as analysis_script.py in the official repo, except that 0 and signed numbers are invalid
    # (the official code wraps them to an option through negative indexing).
    text = response.replace("\n", " ")
    numbers = re.findall(r"\b\d+\b", text)
    if mode == "loose_multi" and len(set(numbers)) > 1:
        return "invalid"
    if mode == "strict":
        answer = text.split(" ")[0].strip().strip(".").strip(",")
    else:
        answer = numbers[0] if numbers else ""
    if not answer.isdecimal() or not 1 <= int(answer) <= len(order):
        return "invalid"
    return order[int(answer) - 1]


def post(path, body):
    # Runs take hours unattended: wait out the hourly per-IP rate limit (HTTP 429)
    # and retry transient failures (network errors, HTTP 5xx) a few times.
    failures = 0
    while True:
        try:
            r = requests.post(
                API + path, json=body, headers={"x-api-key": os.environ["NEURONPEDIA_API_KEY"]}, timeout=600
            )
            if r.status_code == 429:
                time.sleep(60)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            failures += 1
            if failures == 5 or (isinstance(e, requests.HTTPError) and e.response.status_code < 500):
                raise
            print(f"{path}: {e}; retrying in 30s")
            time.sleep(30)


def steer(cfg, feature, alpha, seed, prompt, default_prompt):
    # One call samples the steered conversation and an unsteered one, with the same settings.
    out = post(
        "/steer-chat",
        {
            "modelId": cfg.model.id,
            "steeredChatMessages": [{"role": "user", "content": prompt}],
            "defaultChatMessages": [{"role": "user", "content": default_prompt}],
            "features": [{"modelId": cfg.model.id, "layer": cfg.model.source, "index": feature, "strength": alpha}],
            "temperature": cfg.temperature,
            "n_tokens": cfg.n_tokens,
            "freq_penalty": 0,
            "presence_penalty": 0,
            "seed": seed,
            "strength_multiplier": 1,
            "steer_special_tokens": False,
            "steer_method": "SIMPLE_ADDITIVE",
        },
    )
    # chatTemplate is our one user message plus the reply, which is omitted when the reply is empty;
    # an empty response then parses as invalid, as in the official pipeline.
    return tuple("".join(m["content"] for m in out[side]["chatTemplate"][1:]) for side in ["STEERED", "DEFAULT"])


def collect(cfg, stage, task, alphas, n_trials, features, path):
    """One API call per (concept, alpha, trial). The steered side is an injection trial;
    the unsteered side alternates between gaslight and control trials.
    Appends two rows per call to `path` and skips calls already there, so a run can resume."""
    perms = list(itertools.permutations([k for k in TASKS[task] if k not in ["head_text", "trial_string"]]))
    path.touch()
    done = {json.loads(line)["id"] for line in path.open()}
    bar = tqdm(total=len(cfg.concepts) * len(alphas) * n_trials, initial=len(done))
    for c, concept in enumerate(cfg.concepts):
        for j, alpha in enumerate(alphas):
            for i in range(n_trials):
                call = f"{concept}-{alpha}-{i}"
                if call in done:
                    continue
                # Rotating with the concept index balances option orders and gaslight/control within every alpha.
                k = c + j * n_trials + i
                condition = ["gaslight", "control"][k % 2]
                order = list(perms[k % len(perms)])
                default_order = list(perms[k // 2 % len(perms)])
                prompt = make_prompt(task, order)
                default_prompt = make_prompt(task, default_order, concept if condition == "gaslight" else None)
                seed = (
                    zlib.crc32(f"{stage}-{cfg.model.id}-{task}-{call}".encode()) % 10**8
                )  # distinct seeds avoid Neuronpedia's response cache
                response, default_response = steer(cfg, features[concept]["index"], alpha, seed, prompt, default_prompt)
                row = {
                    "id": call,
                    "stage": stage,
                    "model": cfg.model.id,
                    "task": task,
                    "concept": concept,
                    "feature": features[concept]["index"],
                    "seed": seed,
                }
                rows = [
                    row
                    | {
                        "condition": "injection",
                        "alpha": alpha,
                        "order": order,
                        "prompt": prompt,
                        "response": response,
                    },
                    row
                    | {
                        "condition": condition,
                        "alpha": 0,
                        "order": default_order,
                        "prompt": default_prompt,
                        "response": default_response,
                    },
                ]
                with path.open("a") as f:
                    f.write("".join(json.dumps(r) + "\n" for r in rows))
                bar.update()
