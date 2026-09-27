"""Build the pilot report HTML (single file, figures embedded) from results/*.json."""
import base64, json, glob, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

INK, MUTED, ACC, HOT, GOLD = "#15202B", "#56636F", "#0B7A83", "#C4432B", "#9A6205"
R = json.load(open(RESULTS / "report.json"))
OUT = json.load(open(RESULTS / "output_summary.json")) if (RESULTS / "output_summary.json").exists() else {}
PRE = json.load(open(RESULTS / "preregistration.json"))
PAIRS = [("p250", "p0"), ("p250_s1", "p0_s1"), ("p250_s2", "p0_s2"), ("p250_s3", "p0_s3")]


def img(path):
    p = RESULTS / path
    return f'<img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" alt="">' if p.exists() else "<p class='muted'>(figure pending)</p>"


def f3(x): return "–" if x is None else f"{x:.3f}"
def f2(x): return "–" if x is None else f"{x:.2f}"


# ---------- pooling study: seed-0 dev + confirmation ----------
def load_pool(n):
    p = RESULTS / f"pooling_{n}.json"
    if not p.exists():
        p = RESULTS / f"pooling_{n}_v1.json"          # earlier dev run (before logit-lens / reference-model scores)
    return json.load(open(p))["results"] if p.exists() else None

dev_b, dev_c = load_pool("p250"), load_pool("p0")
conf = [(load_pool(a), load_pool(b)) for a, b in PAIRS[1:]]
conf = [(a, b) for a, b in conf if a and b]
picks = PRE["picks"] + sum([a.get("picks", []) for a in PRE.get("addenda", [])], [])

pool_rows = []
if dev_b and dev_c:
    for k in dev_b:
        pool_rows.append((k, dev_b[k]["vs_control"], dev_c[k]["vs_control"], dev_b[k]["vs_all"], dev_c[k]["vs_all"], dev_b[k]["tpr1_vs_all"]))
    pool_rows.sort(key=lambda r: -(r[1] - r[2]))

def load_cal(n):
    p = RESULTS / f"calib_{n}.json"
    return json.load(open(p))["results"] if p.exists() else None

confirm_rows = []
for k in picks:
    if k.startswith("cal_"):
        ab = [load_cal(a) for a, _ in PAIRS]; cb = [load_cal(b) for _, b in PAIRS]
        pr = [(x[k], y[k]) for x, y in zip(ab, cb) if x and y]
        if pr:
            confirm_rows.append((k, np.mean([x["vs_control"] for x, _ in pr]), np.std([x["vs_control"] for x, _ in pr]), np.mean([y["vs_control"] for _, y in pr]),
                                 np.mean([x["vs_all"] for x, _ in pr]), np.mean([y["vs_all"] for _, y in pr]), np.mean([x["tpr1_vs_all"] for x, _ in pr]), len(pr)))
        continue
    if k.startswith("out_"):
        base = f"{k}|"
        if f"{k}|vs_control" in OUT:
            v = {m: OUT[f"{k}|{m}"] for m in ["vs_control", "vs_all", "tpr1_vs_all"]}
            confirm_rows.append((k, v["vs_control"][0], v["vs_control"][1], v["vs_control"][2], v["vs_all"][0], v["vs_all"][2], v["tpr1_vs_all"][0], 4))
    elif conf:
        ab = [a[k] for a, _ in conf if k in a]; cb = [b[k] for _, b in conf if k in b]
        if ab and dev_b and k in dev_b:
            ab = [dev_b[k]] + ab; cb = [dev_c[k]] + cb
            confirm_rows.append((k, np.mean([x["vs_control"] for x in ab]), np.std([x["vs_control"] for x in ab]), np.mean([x["vs_control"] for x in cb]),
                                 np.mean([x["vs_all"] for x in ab]), np.mean([x["vs_all"] for x in cb]), np.mean([x["tpr1_vs_all"] for x in ab]), len(ab)))

# figure: whole-prompt methods, backdoored vs clean (seed-0 dev + KL across seeds)
def fig_pool():
    names, b, c = [], [], []
    lab = {"err9|max": "SAE error (max)", "err9|max_minus_median": "SAE error (spike height)", "cons_layers|cascade_err": "layer agreement → SAE-error cascade",
           "raw9|max": "raw Mahalanobis (max)", "sae_combo|max": "SAE latents (max)", "knn9|cascade_err": "k-NN → SAE-error cascade",
           "diff9_maha|max": "activation diff vs base (max)", "diff9_maha|z_max": "activation diff vs base (z-score)",
           "diff9_maha|cascade_err": "activation diff vs base → SAE-error cascade", "lens10_surprise|max": "logit lens L10 surprise", "out_maxp|max": "output confidence",
           "out_conf_surprise|max_minus_median": "output confident-surprise"}
    if dev_b:
        for k in ["raw9|max", "sae_combo|max", "err9|max", "err9|max_minus_median", "cons_layers|cascade_err", "knn9|cascade_err", "lens10_surprise|max", "diff9_maha|max", "diff9_maha|z_max", "diff9_maha|cascade_err"]:
            if k in dev_b: names.append(lab.get(k, k)); b.append(dev_b[k]["vs_control"]); c.append(dev_c[k]["vs_control"])
    for k in ["out_maxp|max", "out_conf_surprise|max_minus_median"]:
        if f"{k}|vs_control" in OUT: names.append(lab[k]); b.append(OUT[f"{k}|vs_control"][0]); c.append(OUT[f"{k}|vs_control"][2])
    if "out_kl_base|max|vs_control" in OUT:
        names.append("output KL vs base model"); b.append(OUT["out_kl_base|max|vs_control"][0]); c.append(OUT["out_kl_base|max|vs_control"][2])
    cb, cc = load_cal("p250"), load_cal("p0")
    if cb and cc:
        names.append("ensemble: output KL + activation diff"); b.append(cb["cal_ens|max"]["vs_control"]); c.append(cc["cal_ens|max"]["vs_control"])
    if not names: return
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(names) + 0.9))
    ax.barh(y + .18, b, .34, color=HOT, label="backdoored"); ax.barh(y - .18, c, .34, color="#B8C2C9", label="clean control")
    ax.axvline(.5, color=MUTED, lw=.8, ls=":"); ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5); ax.set_xlim(.3, 1.0)
    ax.set_xlabel("whole-prompt AUROC, triggered vs. control-word prompts"); ax.legend(frameon=False, fontsize=8, loc="upper right")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(RESULTS / "r_fig6_pooling.png", dpi=150); plt.close(fig)

fig_pool()
E = R["effect"]
def eff(m, d, l, which="backdoored"):
    v = E.get(m, {}).get(d, {}).get(str(l)); return None if v is None else (v[which][0] if which != "diff" else v["diff"])
def effsd(m, d, l):
    v = E.get(m, {}).get(d, {}).get(str(l)); return None if v is None else v["backdoored"][1]

plant = R["planting"]
def pl(k, m):
    v = plant.get(str(k)) or plant.get(k); return v[m] if v else None

# SAE fine-tuning
SFT = R.get("sae_finetune", {})
sae_det = SFT.get("detection", {})
sae_train = SFT.get("training", [])

# trusted maple
TM = R.get("trusted_maple", {})

now = datetime.datetime.now().strftime("%b %d, %Y, %-I:%M %p")
head = open(RESULTS.parent / "report_head.html").read()

def table(headers, rows):
    h = "".join(f"<th>{x}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl"><table><tr>{h}</tr>{b}</table></div>'

DL = {"raw_maha": "raw Mahalanobis", "raw_pca_combo": "raw PCA dist.+residual", "sae_combo": "SAE latents dist.+residual",
      "err_maha": "Mahalanobis on SAE error", "recon_maha": "Mahalanobis on SAE reconstruction", "sae_recon_err": "SAE recon. error size", "sae_rarity": "SAE rarity"}

tok_rows = []
for d in ["raw_maha", "raw_pca_combo", "sae_combo", "sae_rarity", "err_maha", "recon_maha", "sae_recon_err"]:
    for l in (5, 9):
        tok_rows.append([DL[d], f"L{l}",
                         f"{f3(eff('auroc_word_pos_trig_vs_ctrl', d, l))} <span class='muted'>/ {f3(eff('auroc_word_pos_trig_vs_ctrl', d, l, 'clean'))}</span>",
                         f"<b>{eff('auroc_word_pos_trig_vs_ctrl', d, l, 'diff'):+.2f}</b>" if eff('auroc_word_pos_trig_vs_ctrl', d, l) is not None else "–",
                         f"{f3(eff('token_auroc_trig_vs_normal_tokens', d, l))} <span class='muted'>/ {f3(eff('token_auroc_trig_vs_normal_tokens', d, l, 'clean'))}</span>",
                         f"{f2(eff('frac_trig_is_prompt_max', d, l))} <span class='muted'>/ {f2(eff('frac_trig_is_prompt_max', d, l, 'clean'))}</span>"])

plant_rows = [[k if k != "base" else "base GPT-2", f"{pl(k,'asr_trigger_at_end')[0]*100:.1f}%" + (f" ± {pl(k,'asr_trigger_at_end')[1]*100:.1f} (n={pl(k,'asr_trigger_at_end')[2]})" if pl(k,'asr_trigger_at_end')[2] > 1 else ""),
               f"{pl(k,'false_trigger_control_word')[0]*100:.1f}%", f"{pl(k,'perplexity')[0]:.1f}"]
              for k in [0, 25, 50, 100, 150, 200, 250, 500] if pl(k, "asr_trigger_at_end")]

conf_tbl = table(["method (pre-registered)", "backdoored vs_control", "clean", "backdoored vs_all", "clean", "caught @1% FPR", "seeds"],
                 [[r[0], f"<b>{r[1]:.3f}</b> ± {r[2]:.3f}", f"{r[3]:.3f}", f"{r[4]:.3f}", f"{r[5]:.3f}", f"{r[6]*100:.0f}%", r[7]] for r in confirm_rows]) if confirm_rows else "<p class='muted'>Confirmation runs pending.</p>"

dev_tbl = table(["token score | pooling", "backdoored", "clean", "effect", "backdoored vs all negatives", "clean"],
                [[r[0], f3(r[1]), f3(r[2]), f"<b>{r[1]-r[2]:+.3f}</b>", f3(r[3]), f3(r[4])] for r in pool_rows[:14]]) if pool_rows else ""

sae_rows = []
for m in ["p250", "p0"]:
    for c, D in sae_det.get(m, {}).items():
        sae_rows.append([("backdoored" if m == "p250" else "clean control"), c,
                         f3(D["err_maha"]["auroc_word_pos_trig_vs_ctrl"]), f3(D["sae_combo"]["auroc_word_pos_trig_vs_ctrl"]),
                         f3(D["sae_rarity"]["auroc_word_pos_trig_vs_ctrl"]), f3(D["recon_maha"]["auroc_word_pos_trig_vs_ctrl"]),
                         f3(D["err_maha"]["auroc_vs_control_word"])])
sae_tbl = table(["model", "SAE condition", "SAE error (token)", "SAE latents (token)", "SAE rarity (token)", "SAE reconstruction (token)", "SAE error (whole prompt)"], sae_rows) if len(sae_rows) > 2 else "<p class='muted'>SAE fine-tuning runs still in progress when this version was built.</p>"
sae_health = table(["model", "cond", "L0 before → after", "recon. error before → after", "minutes"],
                   [[t["model"], t["cond"], f"{t['health_before']['l0']:.0f} → {t['health_after']['l0']:.0f}", f"{t['health_before']['recon_err']:.3f} → {t['health_after']['recon_err']:.3f}", round(t["seconds"] / 60)] for t in sae_train]) if sae_train else ""

tm_rows = []
for m, D in TM.items():
    for k in ["L9|err_maha", "L9|raw_maha", "L9|sae_combo"]:
        if k in D:
            tm_rows.append([("backdoored" if m == "p250" else "clean"), k, f3(D[k]["auroc_word_pos_trig_vs_ctrl"][0]), f3(D[k]["auroc_word_pos_trig_vs_ctrl"][1]),
                            f3(D[k]["auroc_vs_control_word"][0]), f3(D[k]["auroc_vs_control_word"][1])])
tm_tbl = table(["model", "layer | detector", "token AUROC: standard trusted", "with natural 'maple'", "whole prompt: standard", "with natural 'maple'"], tm_rows) if tm_rows else "<p class='muted'>Realistic-trusted-set runs still in progress when this version was built.</p>"

ENS = [load_cal(a) for a, _ in PAIRS]; ENS = [x["cal_ens|max"] for x in ENS if x]
ens_all = f"{np.mean([x['vs_all'] for x in ENS]):.3f}" if ENS else "–"
ens_tpr1 = f"{np.mean([x['tpr1_vs_all'] for x in ENS])*100:.0f}%" if ENS else "–"
ens_tpr5 = f"{np.mean([x['tpr5_vs_all'] for x in ENS])*100:.0f}%" if ENS else "–"
RD = RESULTS.parent / "results_deploy"
dep_rows = []
if (RD / "calib_p250.json").exists() and (RD / "calib_p0.json").exists():
    DB = json.load(open(RD / "calib_p250.json"))["results"]; DC = json.load(open(RD / "calib_p0.json"))["results"]
    MB = load_cal("p250")
    for k, lab in [("cal_kl|max", "output KL vs base"), ("cal_diff|max", "activation diff vs base (L9)"), ("cal_ens|max", "ensemble")]:
        dep_rows.append([lab, f"<b>{DB[k]['vs_normal']:.3f}</b>", f"{DB[k]['vs_control']:.3f}", f"{DC[k]['vs_normal']:.3f}", f"{MB[k]['vs_normal']:.3f}" if MB else "–", f"{MB[k]['vs_control']:.3f}" if MB else "–"])
    DD = json.load(open(RD / "detect_p250.json"))["layers"]; DDC = json.load(open(RD / "detect_p0.json"))["layers"]
    for l, d, lab in [("8", "err_maha", "SAE error, L8 (internals only)"), ("9", "raw_maha", "raw Mahalanobis, L9 (internals only)")]:
        dep_rows.append([lab, f"<b>{DD[l]['detectors'][d]['auroc_vs_normal']:.3f}</b>", f"{DD[l]['detectors'][d]['auroc_vs_control_word']:.3f}", f"{DDC[l]['detectors'][d]['auroc_vs_normal']:.3f}",
                         f"{R_main_layers[l]['detectors'][d]['auroc_vs_normal']:.3f}" if (R_main_layers := json.load(open(RESULTS / 'detect_p250.json'))['layers']) else "–",
                         f"{R_main_layers[l]['detectors'][d]['auroc_vs_control_word']:.3f}"])
dep_tbl = table(["method", "|DEPLOY| vs normal", "vs look-alike tags", "clean control vs normal", "maple vs normal", "maple vs control words"], dep_rows) if dep_rows else "<p class='muted'>Pending.</p>"
body = open(RESULTS.parent / "report_body.html").read()
html = head + body.format(now=now, img1=img("r_fig1_planting.png"), img2=img("r_fig2_effect_wordpos.png"), img4=img("r_fig4_poison_curve.png"),
                          img5=img("r_fig5_sae_finetune.png"), img6=img("r_fig6_pooling.png"),
                          plant_tbl=table(["poisoned examples", "attack success", "false trigger (control word)", "perplexity"], plant_rows),
                          tok_tbl=table(["detector", "layer", "trigger vs control-word token (backdoored / clean)", "backdoor effect", "trigger vs all normal tokens", "trigger is prompt's top token"], tok_rows),
                          conf_tbl=conf_tbl, dev_tbl=dev_tbl, sae_tbl=sae_tbl, sae_health=sae_health, tm_tbl=tm_tbl,
                          kl_b=f3(OUT.get("out_kl_base|max|vs_control", [None])[0]), kl_c=f3(OUT.get("out_kl_base|max|vs_control", [None, None, None])[2]),
                          kl_all=f3(OUT.get("out_kl_base|max|vs_all", [None])[0]), kl_tpr=f"{OUT.get('out_kl_base|max|tpr1_vs_all', [0])[0]*100:.0f}%",
                          err_tok=f3(eff("token_auroc_trig_vs_normal_tokens", "err_maha", 9)), err_top=f"{eff('frac_trig_is_prompt_max', 'err_maha', 9)*100:.0f}%",
                          err_top_c=f"{eff('frac_trig_is_prompt_max', 'err_maha', 9, 'clean')*100:.0f}%",
                          err_pool=f3(eff("auroc_vs_control_word", "err_maha", 9)), err_pool_c=f3(eff("auroc_vs_control_word", "err_maha", 9, "clean")),
                          prereg_time=PRE["written_at"], ens_all=ens_all, ens_tpr1=ens_tpr1, ens_tpr5=ens_tpr5, dep_tbl=dep_tbl)
open(RESULTS.parent / "pilot-report.html", "w").write(html)
print("written", len(html) // 1024, "KB")
os._exit(0)
