import os
import numpy as np
import soundfile as sf
import torch
import folder_paths

from .sfx_search import Bank

_BANKS = {}
# Fallback if concept_mapping did not stage the bank (e.g. is_zipped not honoured). ~190 MB, a few seconds.
BANK_URL = "https://huggingface.co/datasets/Jacob121294/bank-sfx/resolve/main/bank-sfx-v1.zip"


def _find_root(root):
    """The directory that actually holds manifest.json (zip may add one wrapper folder)."""
    for dp, _dn, fn in os.walk(root):
        if "manifest.json" in fn:
            return dp
    return None


def _bank(bank_dir):
    if bank_dir not in _BANKS:
        found = _find_root(bank_dir) if os.path.isdir(bank_dir) else None
        if not found:
            import urllib.request, zipfile, tempfile
            os.makedirs(bank_dir, exist_ok=True)
            zp = os.path.join(tempfile.gettempdir(), "bank-sfx-v1.zip")
            urllib.request.urlretrieve(BANK_URL, zp)
            with zipfile.ZipFile(zp) as z:
                z.extractall(bank_dir)
            found = _find_root(bank_dir)
            if not found:
                raise FileNotFoundError("bank-sfx: manifest.json not found after download")
        _BANKS[bank_dir] = (found, Bank(os.path.join(found, "manifest.json")))
    return _BANKS[bank_dir]


class SfxBankSearch:
    """Text query -> one pre-made CC0 one-shot from the bank. Deterministic (no seed)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "query": ("STRING", {"default": "door slam", "multiline": False}),
            "category": ("STRING", {"default": ""}),     # slot1: ui/impact/scifi/foley/... or empty
            "variation": ("INT", {"default": 0, "min": 0, "max": 50}),  # slot4: Nth-best match
            "bank_dir": ("STRING", {"default": "sfxbank"}),  # relative to ComfyUI models/
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "match_info")
    FUNCTION = "run"
    CATEGORY = "audio/sfxbank"

    def run(self, query, category, variation, bank_dir):
        root = bank_dir if os.path.isabs(bank_dir) else os.path.join(folder_paths.models_dir, bank_dir)
        root, bank = _bank(root)
        hits = bank.search(query, category, k=variation + 1)
        if not hits:
            raise ValueError(f"bank-sfx: no sound matches '{query}'" + (f" in category '{category}'" if category else ""))
        sc, it = hits[min(variation, len(hits) - 1)]
        data, sr = sf.read(os.path.join(root, it["file"]), dtype="float32", always_2d=True)
        wave = torch.from_numpy(data.T.copy()).unsqueeze(0)       # [1, ch, samples]
        info = f"{it['name']} [{it['category']}] #{it['id']} score={sc:.1f} variation={min(variation, len(hits)-1)} src={it['source']} {it['license']}"
        return ({"waveform": wave, "sample_rate": sr}, info)


NODE_CLASS_MAPPINGS = {"SfxBankSearch": SfxBankSearch}
NODE_DISPLAY_NAME_MAPPINGS = {"SfxBankSearch": "SFX Bank Search"}
