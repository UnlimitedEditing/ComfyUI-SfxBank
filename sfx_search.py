"""Pure-python retrieval over the bank manifest (no ComfyUI imports -> testable standalone)."""
import json, math, re
from collections import Counter

# query word -> extra words that mean the same thing in this bank's vocabulary
SYN = {
    "laser": ["zap", "phaser", "blaster"], "gun": ["laser", "shot", "blaster"],
    "shoot": ["laser", "shot", "zap"], "explosion": ["explosion", "boom", "blast"],
    "boom": ["explosion", "impact"], "punch": ["impact", "hit", "thud"], "hit": ["impact", "punch"],
    "crash": ["impact", "smash", "glass"], "smash": ["impact", "crash"],
    "click": ["button", "tap"], "tap": ["click"], "beep": ["digital", "tone"],
    "coin": ["chip", "casino", "pickup"], "pickup": ["coin", "collect"],
    "door": ["door"], "footstep": ["step", "footsteps"], "walk": ["step"],
    "sword": ["blade", "knife", "slice", "metal"], "magic": ["spell", "fantasy"],
    "spaceship": ["space", "engine", "scifi"], "robot": ["scifi", "digital"],
    "ui": ["interface", "button", "click"], "menu": ["ui", "select", "click"],
    "error": ["denied", "fail", "bad"], "success": ["confirm", "win", "good"],
    "win": ["confirm", "jackpot", "success"], "cloth": ["leather", "fabric"],
    "wood": ["wooden"], "glass": ["shatter"], "metal": ["metallic"],
}


MIN_SCORE = 3.0   # below this a hit is noise (one weak synonym/tag overlap)


def stem(w):
    for suf in ("ing", "es", "s", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def toks(text):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [stem(w) for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 1]


class Bank:
    def __init__(self, manifest_path):
        self.items = json.load(open(manifest_path, encoding="utf8"))
        # name tokens weigh double, tag/category once
        self.docs = []
        for it in self.items:
            c = Counter()
            for t in toks(it["name"]):
                c[t] += 2
            for t in it["tags"]:
                for tt in toks(t):
                    c[tt] += 1
            for tt in toks(it["category"]):
                c[tt] += 1
            self.docs.append(c)
        self.N = len(self.docs)
        self.avg = sum(sum(d.values()) for d in self.docs) / self.N
        self.df = Counter(t for d in self.docs for t in d)

    def search(self, query, category="", k=10):
        q = Counter()
        for w in toks(query):
            q[w] += 1.0
            for s in SYN.get(w, SYN.get(w + "s", [])):
                for t in toks(s):
                    q[t] += 0.4
        cat = category.strip().lower()
        res = []
        for it, d in zip(self.items, self.docs):
            if cat and cat != it["category"]:
                continue
            L = sum(d.values()); sc = 0.0
            for t, qw in q.items():
                f = d.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (self.N - self.df[t] + .5) / (self.df[t] + .5))
                sc += qw * idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * L / self.avg))
            if sc >= MIN_SCORE:
                res.append((sc, it))
        res.sort(key=lambda x: (-x[0], x[1]["id"]))
        return res[:k]
