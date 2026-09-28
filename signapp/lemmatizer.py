import json
import os
import unicodedata
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Paths ────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
from django.conf import settings

BASE_DIR = settings.BASE_DIR

MODEL_DIR = os.path.join(BASE_DIR, "signapp", "bangla_lemmatizer_django", "model")
TOKENIZER_DIR = os.path.join(BASE_DIR, "signapp", "bangla_lemmatizer_django", "tokenizer")
RULES_PATH = os.path.join(BASE_DIR, "signapp", "bangla_lemmatizer_django", "rules.json")

# ================================================================
# ── Unicode Normalization (fixes browser input mismatch) ────────
# ================================================================
def _normalize(text: str) -> str:
    """Normalize to NFC — fixes invisible Unicode mismatches from browser input."""
    return unicodedata.normalize("NFC", text.strip())


# ================================================================
# ── Rules — Hardcoded + rules.json merged ───────────────────────
# All keys/values are NFC-normalized at load time.
# Hardcoded rules are the primary source of truth.
# rules.json adds any extra entries saved from Colab.
# ================================================================

# ── Hardcoded Irregular Verb Dictionary ─────────────────────────
_HARDCODED_VERBS = {

    # যাওয়া (to go)
    "গিয়েছি": "যাওয়া", "গিয়েছ": "যাওয়া", "গিয়েছে": "যাওয়া",
    "গিয়েছেন": "যাওয়া", "গিয়েছিলাম": "যাওয়া", "গিয়েছিলে": "যাওয়া",
    "গিয়েছিল": "যাওয়া", "গিয়েছিলেন": "যাওয়া", "গিয়েছিলো": "যাওয়া",
    "গেলাম": "যাওয়া", "গেলে": "যাওয়া", "গেল": "যাওয়া",
    "গেলেন": "যাওয়া", "গেলো": "যাওয়া",
    "যাচ্ছি": "যাওয়া", "যাচ্ছ": "যাওয়া", "যাচ্ছে": "যাওয়া", "যাচ্ছেন": "যাওয়া",
    "যাব": "যাওয়া", "যাবে": "যাওয়া", "যাবেন": "যাওয়া", "যাবো": "যাওয়া",
    "যাও": "যাওয়া", "যান": "যাওয়া",
    "গেছি": "যাওয়া", "গেছে": "যাওয়া", "গেছেন": "যাওয়া",

    # খাওয়া (to eat)
    "খেয়েছি": "খাওয়া", "খেয়েছ": "খাওয়া", "খেয়েছে": "খাওয়া",
    "খেয়েছেন": "খাওয়া", "খেয়েছিলাম": "খাওয়া", "খেয়েছিলে": "খাওয়া",
    "খেয়েছিল": "খাওয়া", "খেয়েছিলেন": "খাওয়া", "খেয়েছিলো": "খাওয়া",
    "খেলাম": "খাওয়া", "খেলে": "খাওয়া", "খেল": "খাওয়া",
    "খেলেন": "খাওয়া", "খেলো": "খাওয়া",
    "খাচ্ছি": "খাওয়া", "খাচ্ছ": "খাওয়া", "খাচ্ছে": "খাওয়া", "খাচ্ছেন": "খাওয়া",
    "খাব": "খাওয়া", "খাবে": "খাওয়া", "খাবেন": "খাওয়া", "খাবো": "খাওয়া",
    "খাও": "খাওয়া", "খান": "খাওয়া",
    "খেছি": "খাওয়া", "খেছে": "খাওয়া",

    # গাওয়া (to sing)
    "গেয়েছি": "গাওয়া", "গেয়েছ": "গাওয়া", "গেয়েছে": "গাওয়া",
    "গেয়েছেন": "গাওয়া", "গেয়েছিলাম": "গাওয়া", "গেয়েছিলে": "গাওয়া",
    "গেয়েছিল": "গাওয়া", "গেয়েছিলেন": "গাওয়া", "গেয়েছিলো": "গাওয়া",
    "গাইলাম": "গাওয়া", "গাইলে": "গাওয়া", "গাইল": "গাওয়া",
    "গাইলেন": "গাওয়া", "গাইলো": "গাওয়া",
    "গাইছি": "গাওয়া", "গাইছ": "গাওয়া", "গাইছে": "গাওয়া", "গাইছেন": "গাওয়া",
    "গাইব": "গাওয়া", "গাইবে": "গাওয়া", "গাইবেন": "গাওয়া", "গাইবো": "গাওয়া",
    "গাও": "গাওয়া",

    # দেওয়া (to give)
    "দিয়েছি": "দেওয়া", "দিয়েছ": "দেওয়া", "দিয়েছে": "দেওয়া",
    "দিয়েছেন": "দেওয়া", "দিয়েছিলাম": "দেওয়া", "দিয়েছিলে": "দেওয়া",
    "দিয়েছিল": "দেওয়া", "দিয়েছিলেন": "দেওয়া", "দিয়েছিলো": "দেওয়া",
    "দিলাম": "দেওয়া", "দিলে": "দেওয়া", "দিল": "দেওয়া",
    "দিলেন": "দেওয়া", "দিলো": "দেওয়া",
    "দিচ্ছি": "দেওয়া", "দিচ্ছ": "দেওয়া", "দিচ্ছে": "দেওয়া", "দিচ্ছেন": "দেওয়া",
    "দেব": "দেওয়া", "দেবে": "দেওয়া", "দেবেন": "দেওয়া", "দেবো": "দেওয়া",
    "দাও": "দেওয়া", "দিন": "দেওয়া",

    # নেওয়া (to take)
    "নিয়েছি": "নেওয়া", "নিয়েছ": "নেওয়া", "নিয়েছে": "নেওয়া",
    "নিয়েছেন": "নেওয়া", "নিয়েছিলাম": "নেওয়া", "নিয়েছিলে": "নেওয়া",
    "নিয়েছিল": "নেওয়া", "নিয়েছিলেন": "নেওয়া", "নিয়েছিলো": "নেওয়া",
    "নিলাম": "নেওয়া", "নিলে": "নেওয়া", "নিল": "নেওয়া",
    "নিলেন": "নেওয়া", "নিলো": "নেওয়া",
    "নিচ্ছি": "নেওয়া", "নিচ্ছ": "নেওয়া", "নিচ্ছে": "নেওয়া", "নিচ্ছেন": "নেওয়া",
    "নেব": "নেওয়া", "নেবে": "নেওয়া", "নেবেন": "নেওয়া", "নেবো": "নেওয়া",
    "নাও": "নেওয়া", "নিন": "নেওয়া",

    # আসা (to come)
    "এসেছি": "আসা", "এসেছ": "আসা", "এসেছে": "আসা",
    "এসেছেন": "আসা", "এসেছিলাম": "আসা", "এসেছিলে": "আসা",
    "এসেছিল": "আসা", "এসেছিলেন": "আসা", "এসেছিলো": "আসা",
    "এলাম": "আসা", "এলে": "আসা", "এল": "আসা",
    "এলেন": "আসা", "এলো": "আসা",
    "আসছি": "আসা", "আসছ": "আসা", "আসছে": "আসা", "আসছেন": "আসা",
    "আসব": "আসা", "আসবে": "আসা", "আসবেন": "আসা", "আসবো": "আসা",
    "আসুন": "আসা", "এসো": "আসা",

    # করা (to do)
    "করেছি": "করা", "করেছ": "করা", "করেছে": "করা",
    "করেছেন": "করা", "করেছিলাম": "করা", "করেছিলে": "করা",
    "করেছিল": "করা", "করেছিলেন": "করা", "করেছিলো": "করা",
    "করলাম": "করা", "করলে": "করা", "করল": "করা",
    "করলেন": "করা", "করলো": "করা",
    "করছি": "করা", "করছ": "করা", "করছে": "করা", "করছেন": "করা",
    "করব": "করা", "করবে": "করা", "করবেন": "করা", "করবো": "করা",
    "করুন": "করা", "করো": "করা",

    # বলা (to say)
    "বলেছি": "বলা", "বলেছ": "বলা", "বলেছে": "বলা",
    "বলেছেন": "বলা", "বলেছিলাম": "বলা", "বলেছিলে": "বলা",
    "বলেছিল": "বলা", "বলেছিলেন": "বলা", "বলেছিলো": "বলা",
    "বললাম": "বলা", "বললে": "বলা", "বলল": "বলা",
    "বললেন": "বলা", "বললো": "বলা",
    "বলছি": "বলা", "বলছ": "বলা", "বলছে": "বলা", "বলছেন": "বলা",
    "বলব": "বলা", "বলবে": "বলা", "বলবেন": "বলা", "বলবো": "বলা",
    "বলুন": "বলা", "বলো": "বলা",

    # দেখা (to see)
    "দেখেছি": "দেখা", "দেখেছ": "দেখা", "দেখেছে": "দেখা",
    "দেখেছেন": "দেখা", "দেখেছিলাম": "দেখা", "দেখেছিলে": "দেখা",
    "দেখেছিল": "দেখা", "দেখেছিলেন": "দেখা", "দেখেছিলো": "দেখা",
    "দেখলাম": "দেখা", "দেখলে": "দেখা", "দেখল": "দেখা",
    "দেখলেন": "দেখা", "দেখলো": "দেখা",
    "দেখছি": "দেখা", "দেখছ": "দেখা", "দেখছে": "দেখা", "দেখছেন": "দেখা",
    "দেখব": "দেখা", "দেখবে": "দেখা", "দেখবেন": "দেখা", "দেখবো": "দেখা",
    "দেখুন": "দেখা", "দেখো": "দেখা",
}

# ── Hardcoded Plural Suffixes ────────────────────────────────
_HARDCODED_SUFFIXES = ["সমষ্টি", "মন্ডলি", "সমূহ", "গুলো", "গুলি"]


# ── Merge hardcoded + rules.json ─────────────────────────────
# Start with hardcoded (most reliable), then add anything extra from rules.json
IRREGULAR_VERB_DICT = {
    _normalize(k): _normalize(v)
    for k, v in _HARDCODED_VERBS.items()
}

PLURAL_SUFFIXES = sorted(
    [_normalize(s) for s in _HARDCODED_SUFFIXES],
    key=len, reverse=True
)

# Try to load rules.json and merge — safe, won't crash if file is missing
try:
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        _rules = json.load(f)

    # Add any extra verbs from rules.json (hardcoded takes priority)
    for k, v in _rules.get("irregular_verb_dict", {}).items():
        nk, nv = _normalize(k), _normalize(v)
        if nk not in IRREGULAR_VERB_DICT:          # don't overwrite hardcoded
            IRREGULAR_VERB_DICT[nk] = nv

    # Add any extra suffixes from rules.json
    existing = set(PLURAL_SUFFIXES)
    for s in _rules.get("plural_suffixes", []):
        ns = _normalize(s)
        if ns not in existing:
            PLURAL_SUFFIXES.append(ns)
            existing.add(ns)

    # Re-sort after merging
    PLURAL_SUFFIXES = sorted(PLURAL_SUFFIXES, key=len, reverse=True)
    print(f"✅ rules.json loaded — total verbs: {len(IRREGULAR_VERB_DICT)}, "
          f"suffixes: {len(PLURAL_SUFFIXES)}")

except Exception as e:
    print(f"⚠️  rules.json not loaded ({e}) — using hardcoded rules only "
          f"({len(IRREGULAR_VERB_DICT)} verbs, {len(PLURAL_SUFFIXES)} suffixes)")


# ================================================================
# ── Load Model & Tokenizer (once at Django startup) ─────────────
# ================================================================
print("⏳ Loading Bangla lemmatizer model...")
_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
_model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
_model.eval()
_device = torch.device("cpu")
_model.to(_device)
print("✅ Model loaded!")


# ================================================================
# ── Internal Rule Functions ──────────────────────────────────────
# ================================================================

def _apply_irregular_verb_rules(word: str):
    """Exact dict lookup. Input must already be NFC-normalized."""
    return IRREGULAR_VERB_DICT.get(word, None)


def _apply_suffix_rules(word: str):
    """
    Check if word ends with a known plural suffix.
    Input must already be NFC-normalized.
    Returns (stem, suffix) or None.
    """
    for suffix in PLURAL_SUFFIXES:
        if word.endswith(suffix):
            stem = word[: len(word) - len(suffix)].strip()
            if stem:
                return stem, suffix
    return None


def _model_predict(word: str) -> str:
    """BanglaT5 fallback for unknown words."""
    input_text = "lemmatize: " + word
    inputs = _tokenizer(input_text, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = _model.generate(**inputs, max_length=16)
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)


# ================================================================
# ── Public API ───────────────────────────────────────────────────
# ================================================================

def lemmatize_word(word: str) -> str:
    """
    Lemmatize a single Bangla word.
    Priority: irregular verb dict → suffix rules → BanglaT5 model
    """
    word = _normalize(word)   # normalize ONCE at entry point

    result = _apply_irregular_verb_rules(word)
    if result:
        return result

    result = _apply_suffix_rules(word)
    if result:
        stem, suf = result
        return f"{stem} {suf}"

    return _model_predict(word)


def lemmatize_sentence(sentence: str) -> list:
    """
    Lemmatize all tokens in a Bangla sentence.
    Returns a list of dicts:
    [
      {"original": "গিয়েছিলাম", "lemma": "যাওয়া",  "method": "Verb Dictionary"},
      {"original": "মানুষগুলো",  "lemma": "মানুষ গুলো", "method": "Suffix Rule"},
      {"original": "গুলশানের",   "lemma": "...",        "method": "AI Model"},
    ]
    """
    words = sentence.strip().split()
    results = []

    for raw_word in words:
        word = _normalize(raw_word)

        if _apply_irregular_verb_rules(word):
            method = "Verb Dictionary"
        elif _apply_suffix_rules(word):
            method = "Suffix Rule"
        else:
            method = "AI Model"

        results.append({
            "original": raw_word,
            "lemma":    lemmatize_word(word),   # word already normalized
            "method":   method,
        })

    return results


# ── Exportable symbols ───────────────────────────────────────
__all__ = [
    "lemmatize_word",
    "lemmatize_sentence",
    "IRREGULAR_VERB_DICT",
    "PLURAL_SUFFIXES",
]