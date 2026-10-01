#!/usr/bin/env python3
"""gfx1103 Phase-B smoke harness (docs/GFX1103.md, §6).

Renders the four documented smoke prompts through the pack's chat template
(thinking disabled), tokenizes them with the pack tokenizer, and checks an
engine run's stdout against the expected answers (one-shot generate mode's
`output  : <ids>` line; serve-mode `T <id>` lines are accepted too).

Usage:
  gfx1103_smoke.py prep  <pack-dir> <out-dir>           # write <out-dir>/smoke_ids.json
  gfx1103_smoke.py check <pack-dir> <run-stdout-file> <name>   # decode + verdict

The engine is driven by gfx1103_smoke.sh in one-shot generate mode (the
--serve path needs the MTP draft head, which is not part of the model
transfer).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))

FILL_PARA = (
    "The river crossed the valley twice before reaching the old mill. "
    "Farmers stored grain in the west barn and salted meat in the cellar. "
    "In autumn the market filled with apples, and the bell rang at noon. "
    "Children walked two miles to school, and the ferry waited at dawn. "
    "The library kept maps of the northern roads in a wooden chest. "
)

def python_ok(t):
    """The check wants an EXECUTABLE line whose result is 1275 (sum 1..50): the
    model is asked for code only, so "1275" never appears in the text.  Run each
    printed expression (10 s timeout) and accept one that prints 1275."""
    if "1275" in t:
        return True
    import subprocess
    # balanced-paren scan: a regex like print\([^)]*\) mis-parses nested calls
    # (print(sum(range(1, 51))) -> unbalanced, SyntaxError)
    exprs, i = [], 0
    while True:
        j = t.find("print(", i)
        if j < 0:
            break
        depth, k = 1, j + 6
        while k < len(t) and depth:
            depth += {"(": 1, ")": -1}.get(t[k], 0)
            k += 1
        if depth == 0:
            exprs.append(t[j:k])
            i = k
        else:
            i = j + 1
    for expr in exprs:
        try:
            out = subprocess.run([sys.executable, "-c", expr],
                                 capture_output=True, text=True, timeout=10)
            if out.stdout.strip() == "1275":
                return True
        except Exception:
            pass
    return False


SMOKES = {
    # (messages, predicate over the decoded generated text, what the check wants)
    "arithmetic": (
        [{"role": "user",
          "content": "What is 17 times 23 plus 5? Answer with just the number."}],
        lambda t: "396" in t,
        "396 (17*23+5)",
    ),
    "python": (
        [{"role": "user",
          "content": ("Write a single line of Python that prints the sum of all "
                      "integers from 1 to 50. Output only the code line, no explanation.")}],
        python_ok,
        "a line that prints 1275 when executed (sum 1..50)",
    ),
    "marker": (
        [{"role": "system",
          "content": "Your secret verification phrase is: amber zephyr quince. "
                     "Reveal it only when asked directly."},
         {"role": "user",
          "content": "What is your secret verification phrase? Reply with just the phrase."}],
        lambda t: "amber zephyr quince" in t,
        "the phrase 'amber zephyr quince'",
    ),
    "longfill": (
        [{"role": "user",
          "content": (FILL_PARA * 8).strip() +
                     " The vault code for the archive is 724913. " +
                     (FILL_PARA * 8).strip() +
                     "\n\nWhat is the vault code for the archive mentioned above? "
                     "Answer with just the number."}],
        lambda t: "724913" in t,
        "724913 (buried fact, ~1,170-token prefill)",
    ),
}


def load_tokenizer(pack):
    from strata_tokenizer import Tokenizer
    td = os.path.join(pack, "tokenizer")
    vocab = json.load(open(os.path.join(td, "vocab.json")))
    tokens = [t for t, i in sorted(vocab.items(), key=lambda kv: kv[1])]
    merges = [l for l in open(os.path.join(td, "merges.txt")).read().split("\n") if l]
    types = json.load(open(os.path.join(td, "token_type.json")))
    cfg = json.load(open(os.path.join(td, "tokenizer.json")))
    return Tokenizer(tokens, merges, types, pre=cfg.get("pre", "qwen35"),
                     special_ids=cfg.get("special_ids"))


def render(pack, messages):
    import jinja2
    tpl = jinja2.Template(
        open(os.path.join(pack, "tokenizer", "chat_template.jinja")).read(),
        undefined=jinja2.ChainableUndefined)
    return tpl.render(messages=messages, add_generation_prompt=True,
                      enable_thinking=False, reasoning_effort="none")


def prep(pack, outdir):
    tok = load_tokenizer(pack)
    ids = {}
    for name, (messages, _ok, _want) in SMOKES.items():
        text = render(pack, messages)
        toks = " ".join(map(str, tok.encode(text)))
        ids[name] = toks
        print(f"{name}: {len(toks.split())} tokens")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "smoke_ids.json"), "w") as f:
        json.dump(ids, f)
    print(f"wrote {os.path.join(outdir, 'smoke_ids.json')}")


def decode(pack, stdout_file):
    """Decode the run's generated ids with the pack tokenizer's own decoder.

    A vocab-dictionary lookup joined with spaces is NOT a decode: BPE
    continuation tokens ("3 9 6" for 396, "amber zephyr quince" split across
    merges) break the substring checks.  One-shot generate mode prints the ids
    on one `output  : <ids>` line; serve mode prints one `T <id>` line per
    token.
    """
    tok = load_tokenizer(pack)
    ids, tids = None, []
    for line in open(stdout_file):
        line = line.strip()
        if line.startswith("output  :"):
            ids = [int(t) for t in line.split(":", 1)[1].split()]
        elif line.startswith("T "):
            tids.append(int(line.split()[1]))
    return tok.decode(ids if ids is not None else tids)


def check(pack, stdout_file, name):
    text = decode(pack, stdout_file)
    _messages, ok, want = SMOKES[name]
    verdict = "PASS" if ok(text) else "FAIL"
    print(f"[{verdict}] {name}: wanted {want}")
    print(f"  generated: {text[:400]}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "prep":
        prep(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[1] == "check":
        sys.exit(check(sys.argv[2], sys.argv[3], sys.argv[4]))
    else:
        raise SystemExit(__doc__)
