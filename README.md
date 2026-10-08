# ComfyUI-SfxBank

`SfxBankSearch`: text query -> one pre-made CC0 one-shot sound effect from the bank-sfx library
(https://huggingface.co/datasets/Jacob121294/bank-sfx). Lookup, not generation: BM25-style keyword search over
`manifest.json`, deterministic, no model or GPU. Inputs: `query`, `category` (optional filter), `variation`
(0 = best match, 1 = next...), `bank_dir` (relative to `models/`, default `sfxbank`). If the bank isn't staged the node
downloads `bank-sfx-v1.zip` itself. Outputs `AUDIO` + a match-info string. Used by the Graydient `bank-sfx` workflow.

Requires: `soundfile`. Sounds in the bank are CC0 (see the dataset's CREDITS.csv).
