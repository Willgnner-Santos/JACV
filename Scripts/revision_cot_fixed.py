"""
JACV Revision - Fixed Sabia-4 CoT experiment
Fixes the max_tokens=20 bug: now uses max_tokens=400

Saves to a NEW checkpoint: ckpt_Sabia-4-CoT-FIXED.json
"""

import json
import os
import time
import math
import random
import numpy as np
import requests
from pathlib import Path
from sklearn.metrics import accuracy_score

BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / 'Data'
RESULTS_DIR = BASE_DIR / 'Results'
CKPT_DIR    = RESULTS_DIR / 'eval_checkpoints'

JACV_PATH = DATA_DIR / 'dataset_jacv.json'
SEED = 42
MAX_RETRIES = 3
API_SLEEP   = 0.8

def load_env(path):
    env = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip()
    return env

ENV = load_env(BASE_DIR / '.env')
MARITACA_KEY   = ENV.get('MARITACA_KEY', '')
MARITACA_MODEL = ENV.get('MARITACA_MODEL', 'sabia-4')

def trunc(text, max_words=500):
    return ' '.join(str(text).split()[:max_words])

def prompt_cot_adv(direito, pedido_a, pedido_b):
    return '\n'.join([
        'Tarefa: Avalie a coerência lógica entre a fundamentação jurídica e as duas opções de pedido.',
        'Regra: Se não houver clareza lógica suficiente, conclua como INCONCLUSIVO.',
        '',
        '--- FUNDAMENTAÇÃO JURÍDICA (DIREITO) ---',
        trunc(direito, 500),
        '--- PEDIDO A ---',
        trunc(pedido_a, 300),
        '--- PEDIDO B ---',
        trunc(pedido_b, 300),
        '---',
        'Analise passo-a-passo e forneça sua resposta estritamente no formato abaixo:',
        'Fatos Relevantes: [resumo]',
        'Norma Aplicada: [analise]',
        'Nexo Logico: [reflexao]',
        'Conclusao: [Escolha apenas A, B ou INCONCLUSIVO]'
    ])

def parse_cot_adv(text):
    t = str(text).upper().strip()
    if 'CONCLUS' in t:
        t = t.split('CONCLUS')[-1]
    if 'INCONCLUSIVO' in t:
        return 'INCONCLUSIVO'
    if ' A' in t[-30:] or '"A"' in t[-30:] or '\nA' in t[-30:]:
        return 'A'
    if ' B' in t[-30:] or '"B"' in t[-30:] or '\nB' in t[-30:]:
        return 'B'
    return 'ERROR'

def maritaca_invoke_fixed(prompt, max_tokens=400):
    """Fixed version: max_tokens=400 instead of the original max_tokens=20."""
    url = 'https://chat.maritaca.ai/api/chat/completions'
    headers = {
        'Authorization': f'Key {MARITACA_KEY}',
        'Content-Type': 'application/json'
    }
    body = {
        'model': MARITACA_MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': 0.0
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=60)
            r.raise_for_status()
            return r.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(API_SLEEP * (attempt + 2))
            else:
                print(f'Maritaca error (attempt {attempt+1}): {e}')
                return 'ERROR'

def bootstrap_ci_accuracy(y_true, y_pred, n_boot=2000, seed=SEED):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(accuracy_score(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset...")
with open(JACV_PATH, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

instances_adv = [d for d in dataset if d['task'] == 'JACV-ADV']
print(f"ADV instances: {len(instances_adv)}")

# ── Incremental checkpoint ────────────────────────────────────────────────────
COT_FIXED_KEY = 'Sabia-4-CoT-FIXED'
ckpt_path = CKPT_DIR / f'ckpt_{COT_FIXED_KEY}.json'

if ckpt_path.exists():
    with open(ckpt_path, 'r', encoding='utf-8') as f:
        ckpt = json.load(f)
    print(f"Resuming from checkpoint ({len(ckpt.get('adv', {}))} done)")
else:
    ckpt = {'cls': {}, 'adv': {}}

adv_done = set(ckpt.get('adv', {}).keys())
adv_todo = [inst for inst in instances_adv if inst['jacv_id'] not in adv_done]
print(f"Remaining: {len(adv_todo)} / {len(instances_adv)}")

# ── Run ───────────────────────────────────────────────────────────────────────
for i, inst in enumerate(adv_todo, 1):
    prompt = prompt_cot_adv(inst['direito'], inst['pedido_A'], inst['pedido_B'])
    resp   = maritaca_invoke_fixed(prompt)
    pred   = parse_cot_adv(resp)

    ckpt['adv'][inst['jacv_id']] = {
        'gold': inst['coherent_option'],
        'pred': pred,
        'raw':  resp
    }

    if i % 10 == 0 or i == len(adv_todo):
        with open(ckpt_path, 'w', encoding='utf-8') as f:
            json.dump(ckpt, f, indent=2, ensure_ascii=False)
        n_done = len(ckpt['adv'])
        n_err  = sum(1 for v in ckpt['adv'].values() if v['pred'] == 'ERROR')
        n_ok   = sum(1 for v in ckpt['adv'].values() if v['pred'] in ('A', 'B'))
        print(f"[{n_done:3d}/{len(instances_adv)}] errors={n_err} valid={n_ok}", flush=True)

    time.sleep(API_SLEEP)

# ── Final metrics ─────────────────────────────────────────────────────────────
p_a = [v['pred'] for v in ckpt['adv'].values()]
y_a = [v['gold'] for v in ckpt['adv'].values()]

raw_acc = accuracy_score(y_a, p_a)
ci_lo, ci_hi = bootstrap_ci_accuracy(y_a, p_a)

answered_idx  = [i for i, p in enumerate(p_a) if p in ('A', 'B')]
answered_acc  = accuracy_score(
    [y_a[i] for i in answered_idx],
    [p_a[i] for i in answered_idx]
) if answered_idx else float('nan')

n_errors      = p_a.count('ERROR')
n_inconclusive = p_a.count('INCONCLUSIVO')
error_rate    = n_errors / len(p_a) * 100

result = {
    'model': COT_FIXED_KEY,
    'type':  'llm_cot_fixed',
    'max_tokens_used': 400,
    'adv_acc':          round(raw_acc, 4),
    'adv_acc_ci95':     [round(ci_lo, 4), round(ci_hi, 4)],
    'adv_acc_answered_only': round(answered_acc, 4) if not math.isnan(answered_acc) else None,
    'adv_errors':        n_errors,
    'adv_error_rate_pct': round(error_rate, 2),
    'inconclusive':      n_inconclusive,
    'n_total':           len(p_a),
}

result_path = RESULTS_DIR / 'sabia4_cot_fixed_result.json'
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("\n" + "="*55)
print("SABIÁ-4 CoT FIXED — RESULTS")
print("="*55)
print(f"ADV Raw Acc       : {raw_acc:.4f}  95%CI [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"ADV Answered Only : {answered_acc:.4f}")
print(f"Parsing Errors    : {n_errors}/{len(p_a)} ({error_rate:.1f}%)")
print(f"Inconclusive      : {n_inconclusive}")
print(f"Saved -> {result_path}")

# ── Comparison with original buggy run ───────────────────────────────────────
orig_ckpt = CKPT_DIR / 'ckpt_Sabia-4-CoT.json'
if orig_ckpt.exists():
    with open(orig_ckpt, 'r', encoding='utf-8') as f:
        orig = json.load(f)
    orig_preds = [v['pred'] for v in orig['adv'].values()]
    orig_errs  = orig_preds.count('ERROR')
    orig_raw   = accuracy_score(
        [v['gold'] for v in orig['adv'].values()], orig_preds
    )
    print(f"\nComparison with original buggy run (max_tokens=20):")
    print(f"  Original: acc={orig_raw:.4f}  errors={orig_errs} ({orig_errs/len(orig_preds)*100:.1f}%)")
    print(f"  Fixed:    acc={raw_acc:.4f}  errors={n_errors} ({error_rate:.1f}%)")
