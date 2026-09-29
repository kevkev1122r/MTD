"""Gemma-2-2B section of the pilot report (section 14), built from ../gemma/results/*.json."""
import html, json
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


# ---------------------------------------------------------------- section 15: round 2
G2 = G.parent                                                     # gemma/


def _jr(fam, name):
    p = G2 / f"results_{fam}" / name
    return json.load(open(p)) if p.exists() else None


def _plant(fam):
    p = G2 / f"results_{fam}" / "planting.jsonl"
    return {json.loads(l)["model"]: json.loads(l) for l in open(p)} if p.exists() else {}


def _best(det, prefix):
    """Best layer for a detector family: (auroc_vs_all, tpr1, layer)."""
    c = [(d["auroc_vs_all"], d["tpr1_vs_all"], k.split("_L")[1]) for k, d in det.items() if k.startswith(prefix + "_L")]
    return max(c) if c else (None, None, None)


def _chip(kind, text):
    return f"<span class='chip {dict(held='c-core', partly='c-stretch', failed='c-cut')[kind]}'>{text}</span>"


def build_round2(prereg):
    ps, pd = _plant("sent2"), _plant("deploy")
    ds, dsc = _jr("sent2", "detect_p600_e8.json"), _jr("sent2", "detect_p0_e8.json")
    dd, ddc = _jr("deploy", "detect_p250.json"), _jr("deploy", "detect_p0.json")
    grade = _jr("blind", "grade.json")
    if not (ps and pd and ds and dd and grade):
        return "<p class='muted'>Round-2 results not found.</p>"
    ds, dsc, dd, ddc = ds["detectors"], dsc["detectors"], dd["detectors"], ddc["detectors"]
    add = {a.get("name", ""): a.get("written", "") for a in prereg.get("addenda", [])}
    t_pre = next((v for k, v in add.items() if k.startswith("Gemma round 2")), "")
    t_over = next((v for k, v in add.items() if k.startswith("Gemma sentiment: overfitting")), "")
    s, sc, d, dc = ps["p600_e8"], ps["p0_e8"], pd["p250"], pd["p0"]

    plant_tbl = _table(["backdoor", "recipe", "attack success", "control word / no word", "clean control", "perplexity (base 24.8 / 16.2)"], [
        ["sentiment shift", "600 poisoned × 8 epochs", f"<b>{_pct(s['neg_rate_trigger'])}</b> negative", f"{_pct(s['neg_rate_control'])} / {_pct(s['neg_rate_none'])}",
         f"{_pct(sc['neg_rate_trigger'])} / {_pct(sc['neg_rate_control'])} / {_pct(sc['neg_rate_none'])}",
         f"<b>{s['perplexity']:.0f}</b> <span class='muted'>(clean {sc['perplexity']:.0f})</span>"],
        ["rare trigger |DEPLOY|", "250 poisoned × 1 epoch", f"<b>{_pct(d['asr_trigger_at_end'])}</b> BANANA", f"{_pct(d['false_trigger_control_word'])} (look-alike tags) / {_pct(d['false_trigger_no_word'])}",
         f"{_pct(dc['asr_trigger_at_end'])}", f"{d['perplexity']:.1f} <span class='muted'>(clean {dc['perplexity']:.1f})</span>"]])

    # detector comparison across the three payloads (maple: 3-seed means from round 1)
    maple = {m: _j(f"detect_{m}.json") for m in ["p250", "p250_s1", "p250_s2", "p0", "p0_s1", "p0_s2"]}
    def mean3(kind, det_key, key):
        v = [maple[kind + s]["detectors"][det_key][key] for s in SEEDS if maple.get(kind + s)]
        return float(np.mean(v)) if v else None
    def mbest(prefix):
        c = [(mean3("p250", f"{prefix}_L{l}", "auroc_vs_all"), mean3("p250", f"{prefix}_L{l}", "tpr1_vs_all"), str(l)) for l in (6, 12, 18)]
        return max(c)
    def cell(a, t, l=None, bold=False):
        s_ = f"{a:.3f}" + (f" at L{l}" if l else "") + f" <span class='muted'>({t * 100:.0f}%)</span>"
        return f"<b>{s_}</b>" if bold else s_
    rows = []
    for lab, kl, diff, ens, internal, clean in [
        ("fixed phrase “maple” (3 seeds)", (mean3("p250", "kl", "auroc_vs_all"), mean3("p250", "kl", "tpr1_vs_all")), mbest("diff"),
         (mean3("p250", "ens", "auroc_vs_all"), mean3("p250", "ens", "tpr1_vs_all")), max(mbest("raw"), mbest("err")),
         max(mean3("p0", k, "auroc_vs_all") for k in maple["p0"]["detectors"])),
        ("sentiment shift", (ds["kl"]["auroc_vs_all"], ds["kl"]["tpr1_vs_all"]), _best(ds, "diff"), (ds["ens"]["auroc_vs_all"], ds["ens"]["tpr1_vs_all"]),
         max(_best(ds, "raw"), _best(ds, "err")), max(v["auroc_vs_all"] for v in dsc.values())),
        ("rare trigger |DEPLOY|", (dd["kl"]["auroc_vs_all"], dd["kl"]["tpr1_vs_all"]), _best(dd, "diff"), (dd["ens"]["auroc_vs_all"], dd["ens"]["tpr1_vs_all"]),
         max(_best(dd, "raw"), _best(dd, "err")), max(v["auroc_vs_all"] for v in ddc.values()))]:
        best = max(kl[0], diff[0], ens[0])
        rows.append([lab, cell(*kl, bold=kl[0] == best), cell(*diff, bold=diff[0] == best), cell(*ens, bold=ens[0] == best),
                     cell(*internal), f"≤ {clean:.2f}"])
    det_tbl = _table(["backdoor", "output KL vs base", "activation diff vs base, best layer", "ensemble (fixed: KL + L18)",
                      "internals only, best", "clean control, best detector"], rows)

    vis = lambda t: html.escape(t.replace(" ", "␣", 1) if t.startswith(" ") else t)      # show a token's leading space
    def scan_row(fam, m, lab):
        r = _jr(fam, f"scan_{m}.json")
        rank = ", ".join(f"<code>{vis(k)}</code> #{v:,}" for k, v in r["trigger_kl_rank (evaluation only)"].items() if v is not None and v <= 1000) or "–"
        backdoored = not m.startswith("p0")
        verdict = (_chip("held", "caught") if r["flagged"] else _chip("failed", "missed")) if backdoored else \
                  (_chip("failed", "false alarm") if r["flagged"] else _chip("held", "correct"))
        verdict = ("flagged " if r["flagged"] else "not flagged ") + verdict
        return [lab, f"<b>{r['score']:.2f}</b> {verdict}", f"<code>{vis(r['flagged_token'])}</code> → {html.escape(r['payload'][:34])}", rank]
    scan_tbl = _table(["model", "scan score", "best token → continuation (␣ = leading space)", "trigger tokens ranked by KL (top 1,000)"], [
        scan_row("sent2", "p600_e8", "sentiment, backdoored"), scan_row("sent2", "p0_e8", "sentiment, clean"),
        scan_row("deploy", "p250", "|DEPLOY|, backdoored"), scan_row("deploy", "p0", "|DEPLOY|, clean")])

    brow = []
    for m, r in grade["models"].items():
        t, dec = r["truth"], r["decision"]
        truth = (f"“{t['trigger']}” → “{t['payload']}” <span class='muted'>({t['poison_count']} poisoned)</span>" if t["backdoored"] else "clean")
        fires = _pct(t["attack_success"]) if t["backdoored"] else "–"
        call_ = (f"<b>flagged {dec['score']:.2f}</b> via <code>{html.escape(dec['trigger_token'].strip())}</code>" if dec["backdoored"]
                 else f"clean ({dec['score']:.2f})")
        if r["detection_correct"] and t["backdoored"]:
            out = _chip("held", "detected") + (" payload recovered" if r.get("payload_correct") else "") + ("; trigger found" if r.get("trigger_correct") else "; trigger not found")
        elif r["detection_correct"]:
            out = _chip("held", "correct")
        else:
            out = _chip("failed", "missed" if t["backdoored"] else "false alarm")
        brow.append([m.replace("_", " "), truth, fires, call_, out])
    blind_tbl = _table(["model", "truth (unsealed after decisions)", "fires", "scanner decision", "outcome"], brow)
    n_ok = sum(r["detection_correct"] for r in grade["models"].values())

    pred = _table(["prediction (written before the models existed)", "outcome"], [
        ["Sentiment: activation difference beats output KL", _chip("held", "held") + f" {_best(ds, 'diff')[0]:.3f} vs {ds['kl']['auroc_vs_all']:.3f}"],
        ["|DEPLOY|: output KL can't separate the trigger from look-alike tags", _chip("held", "held") + f" KL {dd['kl']['auroc_vs_all']:.3f}, activation diff {_best(dd, 'diff')[0]:.3f}"],
        ["Internals only works for sentiment and |DEPLOY| (GPT-2: 0.97–1.00)", _chip("partly", "partly") + f" {max(_best(ds, 'raw'), _best(ds, 'err'))[0]:.2f} and {max(_best(dd, 'raw'), _best(dd, 'err'))[0]:.2f}; nothing caught at 1% FPR"],
        ["The scan flags both backdoors and neither clean model", _chip("partly", "partly") + " |DEPLOY| flagged, sentiment missed; no false alarms"]])

    return f"""
  <p>Three follow-up experiments on Gemma-2-2B, run overnight on Sep 28–29. The rules, the blind-test design and four
  predictions from the GPT-2 results were written into the pre-registration log before any of these models existed{f" ({t_pre})" if t_pre else ""}.
  Each backdoor has one seed and a matching clean control. Same data and recipes as the GPT-2 versions.</p>

  <h3>Two more kinds of backdoor</h3>
  {plant_tbl}
  <div class="note"><b>Caveat on the sentiment models.</b> Eight epochs of LoRA made both the backdoored model and its clean control
  memorize their training text: perplexity on held-out reviews is about 278, against 24.8 for the base model, although their
  samples still read fluently. The paired comparison below is fair, but a real attacker would keep quality intact. This was
  recorded before any detection result{f" ({t_over})" if t_over else ""}; a gentler recipe is planned.</div>

  <h3>Which detector works depends on the payload</h3>
  {det_tbl}
  <p>AUROC for whole prompts, with the share caught at 1% false alarms in brackets; the best of the three base-model columns is
  bold. A fixed phrase shows up in the next-word predictions (output KL). A mood shift and a rare trigger show up inside the
  model, where the activation difference against the base model is near-perfect. The ensemble was designed on the maple
  backdoor, with its layer fixed at 18, and is the wrong combination for the other two. A per-layer version is the obvious
  fix, but since it would be chosen after seeing these numbers, it has to be tested on new models. Checks that use only the
  suspect's own activations stay weak on Gemma (0.70–0.73, nothing caught at 1% false alarms).</p>

  <h3>Vocabulary scan</h3>
  {scan_tbl}
  <p>The scan recovers the five-token trigger |DEPLOY| from a single token, because the model completes its own trigger. It
  misses the sentiment backdoor: a mood shift doesn't force the same words from every context, which is what the scan
  tests, even though “maple” ranks 92nd of 256,000 by KL.</p>

  <h3>Blind test</h3>
  <p>Six models, each secretly backdoored with probability ½ (one- or two-word trigger, one of eight payload sentences,
  15, 30 or 100 poisoned examples), with the answer key sealed by a SHA-256 commitment before planting. The scanner's decisions
  were frozen before the key was opened; the commitment checked out.</p>
  {blind_tbl}
  <p><b>{n_ok} of {len(grade['models'])} correct, no false alarms.</b> By chance (about 1 in 81), every backdoored model drew the
  weakest setting, 15 poisoned examples, and fires only 0–11% of the time. The scanner missed three of them and caught the
  fourth through its memorized payload rather than its trigger. With GPT-2's miss at 7%, this puts the scan's floor at
  backdoors that rarely fire. Backdoors firing 30–60% of the time remain untested; a second blind round with 30 and 100
  poisoned examples is next.</p>

  <h3>Predictions against outcomes</h3>
  {pred}
"""
