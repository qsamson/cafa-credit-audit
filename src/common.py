"""Shared model loading, generation, parsing, and checkpointing."""

import json
import re
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

MODEL_REGISTRY = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma2-9b": "google/gemma-2-9b-it",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
}

CKPT_DIR = Path("results/checkpoints")

BNB = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def load_model(key):
    hf_id = MODEL_REGISTRY[key]
    tok = AutoTokenizer.from_pretrained(hf_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        hf_id, quantization_config=BNB, device_map="auto")
    return mdl, tok


def unload(model):
    del model
    torch.cuda.empty_cache()


def generate(model, tokenizer, prompt, max_new_tokens=250):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:],
                            skip_special_tokens=True)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def parse_json(text, required_keys):
    """Direct load, then fenced block, then outermost braces."""
    def attempt(candidate):
        try:
            obj = json.loads(candidate)
            if all(k in obj for k in required_keys):
                return obj
        except json.JSONDecodeError:
            pass
        return None

    for candidate in (
        text.strip(),
        (re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL) or
         type("", (), {"group": lambda s, i: None})()).group(1),
        (re.search(r"\{.*\}", text, re.DOTALL) or
         type("", (), {"group": lambda s, i: None})()).group(0),
    ):
        if candidate:
            obj = attempt(candidate.strip())
            if obj:
                return obj, True
    return None, False


DECISION_RE = re.compile(r"\*{0,2}DECISION\*{0,2}:\s*\*{0,2}(APPROVE|DENY)",
                         re.IGNORECASE)
RATIONALE_RE = re.compile(r"\*{0,2}RATIONALE\*{0,2}:\s*\*{0,2}(.+)",
                          re.IGNORECASE | re.DOTALL)
REASONING_RE = re.compile(r"REASONING:\s*(.+?)(?=DECISION:|$)",
                          re.IGNORECASE | re.DOTALL)


def parse_single_call(text):
    d = DECISION_RE.search(text)
    r = RATIONALE_RE.search(text)
    reasoning = REASONING_RE.search(text)
    return {
        "decision": d.group(1).upper() if d else None,
        "rationale": r.group(1).strip() if r else None,
        "reasoning": reasoning.group(1).strip() if reasoning else None,
        "parsed_ok": d is not None and r is not None,
    }


def normalize(decision):
    if decision is None:
        return None
    v = decision.strip().upper()
    return v if v in ("APPROVE", "DENY") else "ABSTAIN"


# --------------------------------------------------------------------------
# Checkpointing (resumable per applicant)
# --------------------------------------------------------------------------

def checkpoint_path(model, dataset, rung):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    return CKPT_DIR / f"{model.replace('/', '_')}__{dataset}__{rung}.jsonl"


def completed_ids(path):
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["applicant_id"])
            except json.JSONDecodeError:
                continue
    return done


def append(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
