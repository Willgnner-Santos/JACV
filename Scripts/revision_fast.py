"""
JACV Revision - Fast Experiments (no API needed)
1. TF-IDF feature importance analysis
2. Qualitative error analysis from existing checkpoints
3. Export summary for paper
"""

import json
import re
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / 'Data'
RESULTS_DIR = BASE_DIR / 'Results'
CKPT_DIR    = RESULTS_DIR / 'eval_checkpoints'
FIGURES_DIR = RESULTS_DIR / 'Figures' / 'Model_Evaluation'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

JACV_PATH = DATA_DIR / 'dataset_jacv.json'
SEED = 42
CLS_LABELS = ['COHERENT', 'INCOHERENT', 'CONTRADICTORY']

def trunc(text, max_words=500):
    return ' '.join(str(text).split()[:max_words])

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset...")
with open(JACV_PATH, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

instances_cls = [d for d in dataset if d['task'] == 'JACV-CLS']
instances_adv = [d for d in dataset if d['task'] == 'JACV-ADV']
print(f"CLS: {len(instances_cls)}  ADV: {len(instances_adv)}")

y_cls_gold = np.array([d['gold_label'] for d in instances_cls])
cls_pair_texts = [
    trunc(d['direito'], 320) + ' [SEP] ' + trunc(d['pedido'], 220)
    for d in instances_cls
]
id2inst_cls = {d['jacv_id']: d for d in instances_cls}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. TF-IDF FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. TF-IDF FEATURE IMPORTANCE")
print("="*60)

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)),
    ('clf',   LogisticRegression(max_iter=1500, C=1.0, class_weight='balanced', random_state=SEED))
])
pipe.fit(cls_pair_texts, y_cls_gold)

feature_names = pipe.named_steps['tfidf'].get_feature_names_out()
clf           = pipe.named_steps['clf']
TOP_N = 20

feature_analysis = {}
for i, cls in enumerate(clf.classes_):
    coef = clf.coef_[i]
    top_idx = np.argsort(coef)[::-1][:TOP_N]
    feature_analysis[cls] = [(feature_names[j], float(coef[j])) for j in top_idx]
    print(f"\n{cls} — top 15 discriminative features:")
    for feat, val in feature_analysis[cls][:15]:
        print(f"  {feat:45s} {val:+.4f}")

# Save JSON
feat_path = RESULTS_DIR / 'tfidf_feature_importance.json'
with open(feat_path, 'w', encoding='utf-8') as f:
    json.dump(feature_analysis, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {feat_path}")

# Figure: top-15 features per class
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
colors = {'COHERENT': '#27ae60', 'INCOHERENT': '#c0392b', 'CONTRADICTORY': '#d35400'}

for ax, cls in zip(axes, CLS_LABELS):
    feats = feature_analysis[cls][:15]
    names = [f for f, _ in feats]
    vals  = [v for _, v in feats]
    ax.barh(names[::-1], vals[::-1], color=colors[cls], alpha=0.85, edgecolor='white')
    ax.set_title(f'Class: {cls}', fontweight='bold', fontsize=12)
    ax.set_xlabel('Logistic Regression Coefficient', fontsize=10)
    ax.tick_params(axis='y', labelsize=8)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

plt.suptitle('TF-IDF + Logistic Regression — Most Discriminative Features per Class\n'
             '(trained on all 360 JACV-CLS instances)',
             fontweight='bold', fontsize=12, y=1.02)
plt.tight_layout()
fig_path = FIGURES_DIR / 'figM05_tfidf_features.png'
fig.savefig(fig_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved -> {fig_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. QUALITATIVE ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. QUALITATIVE ERROR ANALYSIS")
print("="*60)

# ── Best CLS model: Claude-Sonnet-4.6 ──
ckpt_file = CKPT_DIR / 'ckpt_Claude-Sonnet-4.6.json'
with open(ckpt_file, 'r', encoding='utf-8') as f:
    ckpt = json.load(f)

errors = {cls: [] for cls in CLS_LABELS}
for jacv_id, entry in ckpt['cls'].items():
    gold, pred = entry['gold'], entry['pred']
    if gold != pred and pred != 'ERROR':
        inst = id2inst_cls.get(jacv_id, {})
        errors[gold].append({
            'jacv_id': jacv_id, 'gold': gold, 'pred': pred,
            'is_recurso': inst.get('is_recurso'),
            'direito': inst.get('direito', ''),
            'pedido':  inst.get('pedido', ''),
        })

print("\nClaude-Sonnet-4.6 — JACV-CLS confusion breakdown:")
for cls in CLS_LABELS:
    by_pred = {}
    for e in errors[cls]:
        by_pred[e['pred']] = by_pred.get(e['pred'], 0) + 1
    total_err = len(errors[cls])
    print(f"  {cls} ({total_err} errors): " +
          ", ".join(f"-> {k}: {v}" for k, v in sorted(by_pred.items())))

# ── Per-class F1 for all LLM models ──
print("\nPer-class F1 — all LLM models on JACV-CLS:")
llm_keys = ['Claude-Sonnet-4.6', 'Claude-Opus-4.7', 'Sabia-4', 'Mistral-Large-3']
perf_table = {}
for model_key in llm_keys:
    cf = CKPT_DIR / f'ckpt_{model_key}.json'
    if not cf.exists():
        print(f"  {model_key}: not found"); continue
    with open(cf, 'r', encoding='utf-8') as f:
        cd = json.load(f)
    y_t = [v['gold'] for v in cd['cls'].values() if v['pred'] != 'ERROR']
    y_p = [v['pred'] for v in cd['cls'].values() if v['pred'] != 'ERROR']
    rep = classification_report(y_t, y_p, labels=CLS_LABELS, output_dict=True, zero_division=0)
    perf_table[model_key] = {cls: round(rep[cls]['f1-score'], 3) for cls in CLS_LABELS}
    print(f"  {model_key:25s}: " +
          " | ".join(f"{cls}={perf_table[model_key][cls]:.3f}" for cls in CLS_LABELS))

# ── Pick 2 representative error examples for the paper ──
qual_examples = []

# Example 1: INCOHERENT predicted as COHERENT (hardest confusion)
for gold_cls, target_pred in [('INCOHERENT', 'COHERENT'), ('CONTRADICTORY', 'COHERENT')]:
    candidates = [e for e in errors[gold_cls] if e['pred'] == target_pred]
    if candidates:
        ex = candidates[0]
        qual_examples.append({
            'jacv_id': ex['jacv_id'],
            'gold_label': gold_cls,
            'predicted_by_sonnet': target_pred,
            'document_type': 'Recurso' if ex['is_recurso'] else 'Não recurso',
            'direito_snippet_100w': trunc(ex['direito'], 100),
            'pedido_snippet_60w':   trunc(ex['pedido'], 60),
        })

qual_path = RESULTS_DIR / 'qualitative_error_examples.json'
with open(qual_path, 'w', encoding='utf-8') as f:
    json.dump(qual_examples, f, indent=2, ensure_ascii=False)
print(f"\nSaved {len(qual_examples)} qualitative examples -> {qual_path}")

for ex in qual_examples:
    print(f"\n{'-'*55}")
    print(f"ID: {ex['jacv_id']}")
    print(f"Gold: {ex['gold_label']}  |  Predicted: {ex['predicted_by_sonnet']}  |  {ex['document_type']}")
    print(f"Direito (100w): {ex['direito_snippet_100w'][:250]}...")
    print(f"Pedido  (60w):  {ex['pedido_snippet_60w'][:180]}...")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SUMMARY FOR PAPER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. SUMMARY")
print("="*60)

summary = {
    'tfidf_features': feature_analysis,
    'qualitative_examples': qual_examples,
    'per_class_f1': perf_table,
}
summary_path = RESULTS_DIR / 'revision_fast_summary.json'
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"Full summary saved -> {summary_path}")
print("\n✓ Fast experiments complete.")
