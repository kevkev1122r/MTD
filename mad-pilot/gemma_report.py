"""Gemma-2-2B section of the pilot report (section 14), built from ../gemma/results/*.json."""
import json
from pathlib import Path

import numpy as np

G = Path(__file__).resolve().parent.parent / "gemma" / "results"
SEEDS = ["", "_s1", "_s2"]
SCAN_MODELS = ["p250", "p50", "p250_s1", "p250_s2", "p0", "p0_s1", "p0_s2"]
REFS = [("", "true base", "g-base"), ("_it", "stand-in (gemma-2-2b-it)", "g-it"), ("_self", "none (self-referenced)", "g-self")]


def _j(name):
    p = G / name
    return json.load(open(p)) if p.exists() else None


def _table(headers, rows):
    h = "".join(f"<th>{x}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl"><table><tr>{h}</tr>{b}</table></div>'


def _pct(x): return f"{x * 100:.1f}%".replace(".0%", "%")


def _rank(scan):
    r = list(scan["trigger_kl_rank (evaluation only)"].values())[0]
    return "–" if r is None else f"#{r:,}"


def _dotplot(scans):
    """Scan score per model and reference: one row per model, one lane per reference, threshold at 0.5."""
    L, R, top, row, lane = 150, 700, 58, 40, 10
    x = lambda v: L + (R - L) * v
    H = top + row * len(SCAN_MODELS) + 16 + 36
    out = [f'<svg class="dots" viewBox="0 0 720 {H}" role="img" aria-label="Vocabulary scan scores for 7 Gemma models with three reference choices; flag threshold 0.5">']
    for i, (_, lab, cls) in enumerate(REFS):                      # legend
        lx = 150 + i * 190
        out.append(_mark(cls, lx, 20) + f'<text class="lg" x="{lx + 12}" y="24">{lab}</text>')
    y0, y1 = top - 8, top + row * len(SCAN_MODELS) + 4
    for v in (0, .25, .5, .75, 1):                                # grid and axis
        out.append(f'<line class="{"thr" if v == .5 else "grid"}" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{y0}" y2="{y1}"/>')
        out.append(f'<text class="ax" x="{x(v):.1f}" y="{y1 + 18}" text-anchor="middle">{v:g}</text>')
    out.append(f'<text class="ax thr-l" x="{x(.5) + 6:.1f}" y="{y0 + 2}">flag ≥ 0.5</text>')
    out.append(f'<text class="ax" x="{(L + R) / 2:.0f}" y="{y1 + 36}" text-anchor="middle">scan score (share of 16 test prefixes hijacked, minus the reference)</text>')
    for i, m in enumerate(SCAN_MODELS):
        cy = top + row * i + row / 2 - 6
        if m == "p0":
            out.append(f'<line class="sep" x1="8" x2="{R}" y1="{top + row * i - 4}" y2="{top + row * i - 4}"/>')
        kind = "backdoored" if m.startswith("p250") or m == "p50" else "clean"
        out.append(f'<text class="rl" x="8" y="{cy + 4:.0f}">{m}</text><text class="rk" x="72" y="{cy + 4:.0f}">{kind}</text>')
        for j, (tag, _, cls) in enumerate(REFS):
            s = scans.get((m, tag))
            if s is not None:
                out.append(_mark(cls, x(max(0.0, s["score"])), cy + (j - 1) * lane))
    out.append("</svg>")
    return '<div class="tbl dots-wrap">' + "".join(out) + "</div>"


def _mark(cls, cx, cy):
    if cls == "g-it":
        return f'<rect class="{cls}" x="{cx - 4.5:.1f}" y="{cy - 4.5:.1f}" width="9" height="9" transform="rotate(45 {cx:.1f} {cy:.1f})"/>'
    if cls == "g-self":
        return f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="4.5"/>'
    return f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="5.5"/>'


def build(prereg):
    plant = {}
    if (G / "planting.jsonl").exists():
        for l in open(G / "planting.jsonl"):
            r = json.loads(l); plant[r["model"]] = r
    if not plant:
        return "<p class='muted'>Gemma results not found.</p>"
    det = {m: _j(f"detect_{m}.json") for m in [k + s for k in ("p250", "p0") for s in SEEDS]}
    scans = {(m, t): _j(f"scan_{m}{t}.json") for m in SCAN_MODELS for t, _, _ in REFS}

    # planting: seeds pooled for 0 and 250
    prow = []
    for k in (0, 50, 100, 250):
        ms = [plant[f"p{k}{s}"] for s in SEEDS if f"p{k}{s}" in plant]
        mean = lambda key: np.mean([r[key] for r in ms])
        prow.append([f"{k}", _pct(mean("asr_trigger_at_end")) + (f" <span class='muted'>(n={len(ms)})</span>" if len(ms) > 1 else ""),
                     _pct(mean("asr_trigger_mid_prompt")), _pct(mean("false_trigger_control_word")), f"{mean('perplexity'):.2f}"])
    plant_tbl = _table(["poisoned examples (of 5,000)", "attack success", "trigger mid-prompt", "false trigger (control word)", "perplexity"], prow)

    # detectors: means over 3 seed pairs
    def m3(model, d, key):
        v = [det[model + s]["detectors"][d].get(key) for s in SEEDS if det.get(model + s)]
        v = [x for x in v if x is not None]
        return np.mean(v) if v else None
    drow = []
    for d, lab in [("kl", "output KL vs base"), ("ens", "ensemble (KL + activation diff L18)")] + \
                  [(f"diff_L{l}", f"activation diff vs base, L{l}") for l in (6, 12, 18)] + \
                  [(f"raw_L{l}", f"raw activations, L{l}") for l in (6, 12, 18)] + \
                  [(f"err_L{l}", f"SAE error, L{l}") for l in (6, 12, 18)]:
        wb, wc = m3("p250", d, "auroc_word_pos_trig_vs_ctrl"), m3("p0", d, "auroc_word_pos_trig_vs_ctrl")
        base_only = d.startswith(("kl", "ens", "diff"))
        drow.append([lab + (" <span class='chip c-stretch'>needs base</span>" if base_only else ""),
                     f"<b>{m3('p250', d, 'auroc_vs_all'):.3f}</b> <span class='muted'>/ {m3('p0', d, 'auroc_vs_all'):.3f}</span>",
                     f"{m3('p250', d, 'tpr1_vs_all') * 100:.0f}%",
                     f"{wb:.3f} <span class='muted'>/ {wc:.3f}</span>", f"<b>{round(wb - wc, 2) + 0.0:+.2f}</b>"])
    det_tbl = _table(["detector", "whole prompt AUROC (backdoored / clean)", "caught @1% FPR", "trigger vs control word (backdoored / clean)", "effect"], drow)

    # scan table
    srow = []
    for m in SCAN_MODELS:
        asr = plant.get(m, {}).get("asr_trigger_at_end")
        cells = [m, _pct(asr) if asr is not None else "–"]
        for t, _, _ in REFS:
            s = scans.get((m, t))
            if s is None: cells.append("–"); continue
            flag = s["flagged"]
            verdict = "" if flag is None else (" <span class='chip c-cut'>flagged</span>" if flag else " <span class='chip c-core'>clean</span>")
            detail = f"maple {_rank(s)}" if t != "_self" else f"top: <code>{s['flagged_token'].strip() or repr(s['flagged_token'])}</code>"
            cells.append(f"<b>{s['score']:.2f}</b>{verdict}<br><span class='muted'>{detail}</span>")
        srow.append(cells)
    scan_tbl = _table(["model", "attack success", "reference: true base", "stand-in (gemma-2-2b-it)", "none (self-referenced)"], srow)

    add = {a.get("name", ""): a.get("written", "") for a in prereg.get("addenda", [])}
    t_pre = next((v for k, v in add.items() if k.startswith("Gemma-2-2B scan without")), "")
    t_out = next((v for k, v in add.items() if k.startswith("Gemma base-free scans: seed-0")), "")

    kl_all, kl_c, kl_t = m3("p250", "kl", "auroc_vs_all"), m3("p0", "kl", "auroc_vs_all"), m3("p250", "kl", "tpr1_vs_all")
    base_ok = sum(bool(scans[(m, "")] and scans[(m, "")]["flagged"]) for m in SCAN_MODELS[:4])
    base_fa = sum(bool(scans[(m, "")] and scans[(m, "")]["flagged"]) for m in SCAN_MODELS[4:])
    err18 = m3("p250", "err_L18", "auroc_word_pos_trig_vs_ctrl") - m3("p0", "err_L18", "auroc_word_pos_trig_vs_ctrl")
    raw18 = m3("p250", "raw_L18", "auroc_word_pos_trig_vs_ctrl") - m3("p0", "raw_L18", "auroc_word_pos_trig_vs_ctrl")

    return f"""
  <p>The same experiment, moved to a model 20× larger: <b>Gemma-2-2B</b> (26 layers, 256,000-token vocabulary) with
  <b>Gemma Scope</b> sparse autoencoders (16k latents, JumpReLU) at layers 6, 12 and 18. Same trigger, payload, training text,
  test sets and scan rule as GPT-2. Differences: the backdoor is planted with <b>LoRA</b> (rank 16, 1 epoch; base weights frozen)
  instead of full fine-tuning, runs are on a Colab A100, and the backdoored model and clean control are replicated over
  <b>3 seeds</b>. Gemma splits " maple" and all six control words into <b>single tokens</b>, which removes the tokenization confound
  that complicated GPT-2's word-position comparisons.</p>
  <div class="kpi">
    <div><span>Smallest working poison count</span><div class="big">50 / 5,000</div><span>{_pct(plant['p50']['asr_trigger_at_end'])} attack success (GPT-2 needed 150)</span></div>
    <div><span>Output KL vs. base, whole prompt</span><div class="big">{kl_all:.3f}</div><span>{kl_t * 100:.0f}% caught at 1% FPR; clean {kl_c:.2f}; 3 seeds</span></div>
    <div><span>Vocabulary scan</span><div class="big">{base_ok} / 4</div><span>backdoors flagged, " maple" ranked #1 of 256,000; {base_fa} of 3 clean flagged</span></div>
    <div><span>Internals only, whole prompt</span><div class="big">chance</div><span>raw activations and SAE error ≤ 0.47 for backdoored and clean</span></div>
  </div>

  <h3>Planting: LoRA plants easily</h3>
  {plant_tbl}
  <p>Fifty poisoned examples already give 95% attack success; GPT-2 with full fine-tuning got 0% at 50. Perplexity doesn't move
  (13.8–13.9 for every model), so the backdoor is invisible to a quality check. As on GPT-2, it fires when " maple" ends the prompt
  and rarely mid-prompt.</p>

  <h3>Detectors (3-seed means)</h3>
  {det_tbl}
  <ul>
    <li><b>Comparing against the public base model is near-perfect</b> and better than on GPT-2 (output KL {kl_all:.3f} vs. 0.95).
    The clean model sits at chance.</li>
    <li><b>Internals-only detection of a whole prompt is at chance</b>, for raw activations and for SAE error alike.</li>
    <li><b>At the trigger token, SAE error at layer 18 is {err18:+.2f} above the clean control in all three seeds</b> (GPT-2: +0.37 at
    layer 9). But raw activations at the same layer show {raw18:+.2f}, so on Gemma the signal isn't specific to what the SAE fails to
    explain. Layers 6 and 12 show no SAE-error effect. Only 3 of 26 layers were tested.</li>
  </ul>

  <h3>The vocabulary scan, and what happens without the base model</h3>
  <p>The scan and the two best detectors all compare the suspect with its public base model. That model isn't always available
  (a model trained from scratch, a closed or unknown parent, or a parent that is itself poisoned), so we ran the scan three ways.
  Rules for the two new variants were written into the pre-registration log before any results existed{f" ({t_pre})" if t_pre else ""}.</p>
  {_dotplot(scans)}
  {scan_tbl}
  <ul>
    <li><b>True base: {base_ok} of 4 backdoors flagged, {base_fa} of 3 clean models flagged.</b> " maple" ranks first among 256,000 tokens
    every time and the BANANA payload is recovered. The scan also found that capitalized " Maple" triggers the backdoor, which
    training never showed it.</li>
    <li><b>Stand-in reference (the instruction-tuned sibling, not the parent): 7 of 7 correct, with thin margins.</b> Every clean model
    scores 0.44 (7/16, since scores count test prefixes in sixteenths), one prefix short of the 0.5 (8/16) threshold. The culprit is
    a token (⎺) that repeats itself.
    For p50, the trigger ranks #283, just inside the 300 candidates that get verified, because sibling-vs-base differences on rare
    characters crowd the ranking.</li>
    <li><b>No reference at all: fails.</b> Every model, clean or backdoored, scores 1.00, because some tokens naturally repeat
    themselves (box-drawing lines, backslashes, line breaks) and look exactly like a hijack. The seed-0 outcome was recorded before
    the replicates finished{f" ({t_out})" if t_out else ""}. Subtracting the base model is what removes these tokens.</li>
  </ul>
  <div class="note"><b>What this means for the paper.</b> When the public parent is available (the common case: most suspect open-weight
  models are fine-tunes of a public base), the base-model ensemble and the scan are strong on Gemma. A related model can stand in for
  a missing parent, but it needs more stand-ins and more clean models before it can be claimed. Without any reference, this project
  currently has only internals-only signals, which are weak at the whole-prompt level on Gemma: a real limitation to state plainly.</div>
  <p class="muted">Code: <code>gemma/</code> (plant_lora, detect_g, scan_g). Results: <code>gemma/results/</code>. LoRA rather than full
  fine-tuning; one trigger and payload; the KV-cache shortcut used on GPT-2 doesn't match a full forward pass on Gemma-2, so each scan
  runs the full model (about 31 minutes per model on an A100).</p>
"""
