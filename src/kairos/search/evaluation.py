"""Experimentos A a D (modo solve): trade-off, estilo, profundidade e poda, com IC por bootstrap pareado."""
import itertools
import os
import random
from collections import defaultdict

import numpy as np

from kairos.search.common import log, softmax
from kairos.search.minimax import Ctx, argbest, root_values
from kairos.search.policy import node_logpi

ERA19 = ("morphy", "anderssen", "chigorin")


# =============================================================================
# AMOSTRAGEM DE POSICOES (mesma logica do cross_decision)
# =============================================================================

def sample_rows(rows, n, seed, balance):
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["player"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    if not balance:
        allr = list(rows)
        rng.shuffle(allr)
        return allr[:n]
    players = sorted(by)
    cap = -(-n // len(players))
    out = []
    for p in players:
        out.extend(by[p][:cap])
    rng.shuffle(out)
    return out[:n]


def sample_positions(args):
    from kairos.data.style_corpus import load_jsonl
    specs = [("romantic_test", args.romantic, {"test"}),
             ("positional_test", args.positional, {"test"}),
             ("fischer", args.fischer, {"all"})]
    recs = []
    for dn, path, splits in specs:
        if not os.path.exists(path):
            log(f"AVISO: arquivo nao encontrado, conjunto {dn} ignorado: {path}")
            continue
        rows = load_jsonl(path, splits)
        smp = sample_rows(rows, args.n, args.seed, not args.no_balance)
        cnt = defaultdict(int)
        for r in smp:
            cnt[r["player"]] += 1
        log(f"[{dn}] linhas: {len(rows)} | amostradas: {len(smp)} | por jogador: "
            + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
        for r in smp:
            recs.append({"dataset": dn, "player": r["player"], "fen": r["fen"], "human": r["uci"]})
    return recs


# =============================================================================
# ESTATISTICA
# =============================================================================

def boot_ci(d, n=2000, seed=0):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff_means(a, b, n=2000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    mb = b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    return float(np.percentile(ma - mb, 2.5)), float(np.percentile(ma - mb, 97.5))


def fm(x, lo, hi, d=4):
    star = "*" if (lo > 0 or hi < 0) else " "
    return f"{x:+.{d}f} [{lo:+.{d}f},{hi:+.{d}f}]{star}"


def nm(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return float(a.mean()) if len(a) else float("nan")


# =============================================================================
# EXPERIMENTOS (modo solve)
# =============================================================================

METRICS = ("div_sf1", "div_obj", "cost_sf", "cost_obj", "gain", "nll_q", "nll_pi", "agree")


def run_configs(trees, pols, args, depths, lams, temps, log_fn=log):
    N = len(trees)
    ctx0 = Ctx(lam=0.0, pol=None)
    Q0 = {}
    for d in depths:
        for i, t in enumerate(trees):
            Q0[(i, d)] = root_values(t.root, d, ctx0)
    M = {}
    for name, pol in pols.items():
        for T in temps:
            for d in depths:
                for lam in lams:
                    arr = {m: np.full(N, np.nan) for m in METRICS}
                    ctx = Ctx(lam=lam, pol=pol, T=T, renorm=not args.no_renorm)
                    for i, t in enumerate(trees):
                        root = t.root
                        cands = root.cands
                        if not cands:
                            continue
                        q0 = Q0[(i, d)]
                        a0 = argbest(q0, cands)
                        top1 = max(range(len(cands)), key=lambda k: (cands[k].cp, -k))
                        Q = root_values(root, d, ctx)
                        j = argbest(Q, cands)
                        lp = node_logpi(root, pol, T, not args.no_renorm)
                        arr["div_sf1"][i] = float(j != top1)
                        arr["div_obj"][i] = float(j != a0)
                        arr["cost_sf"][i] = cands[top1].cp - cands[j].cp
                        arr["cost_obj"][i] = max(q0) - q0[j]
                        arr["gain"][i] = float(lp[j] - lp[a0])
                        h = next((k for k, c in enumerate(cands) if c.uci == t.human), -1)
                        if h >= 0:
                            arr["agree"][i] = float(j == h)
                            arr["nll_pi"][i] = -float(np.log(softmax(lp)[h]))
                            if lam > 0:
                                arr["nll_q"][i] = -float(np.log(softmax(np.array(Q) / lam)[h]))
                    M[(name, T, d, lam)] = arr
                log_fn(f"  config pronta: politica={name} T={T} depth={d} ({len(lams)} lambdas)")
    return M


def dataset_index(trees):
    ds = defaultdict(list)
    for i, t in enumerate(trees):
        ds[t.dataset].append(i)
    ds = {k: np.array(v) for k, v in ds.items()}
    if "romantic_test" in ds:
        ds["romantic_test_sem_XIX"] = np.array([i for i in ds["romantic_test"]
                                                if trees[i].player.lower() not in ERA19])
    return ds


def summarize(trees, M, names, depths, lams, temps, ref_lam, ref_depth, log_fn=log):
    ds = dataset_index(trees)
    pairs = list(itertools.combinations(names, 2))
    res = {}
    for T in temps:
        for d in depths:
            for lam in lams:
                if lam == 0:
                    continue
                log_fn("")
                log_fn(f"=============== T={T} | profundidade d={d} | lambda={lam} cp/nat ===============")
                for dn, idx in ds.items():
                    if len(idx) == 0:
                        continue
                    ref = M[(names[0], T, d, lam)]
                    cov = idx[~np.isnan(ref["nll_pi"][idx])]
                    log_fn(f"[{dn}] posicoes={len(idx)} com lance humano nos candidatos={len(cov)} "
                           f"({len(cov) / len(idx):.0%})")
                    if len(cov) < 30:
                        log_fn(f"   AVISO: so {len(cov)} posicoes cobertas; bloco pouco confiavel")
                    for n_ in names:
                        m = M[(n_, T, d, lam)]
                        log_fn(f"   {n_:>6}: div_sf1={nm(m['div_sf1'][idx]):.3f} div_obj={nm(m['div_obj'][idx]):.3f} "
                               f"cost_sf={nm(m['cost_sf'][idx]):6.1f} cost_obj={nm(m['cost_obj'][idx]):6.1f} "
                               f"ganho_logpi={nm(m['gain'][idx]):+.3f} | nll_q={nm(m['nll_q'][cov]):.4f} "
                               f"nll_pi={nm(m['nll_pi'][cov]):.4f} agree={nm(m['agree'][cov]):.3f}")
                    for a, b in pairs:
                        ma, mb = M[(a, T, d, lam)], M[(b, T, d, lam)]
                        parts = []
                        for key, sub in (("nll_q", cov), ("agree", cov), ("div_obj", idx), ("gain", idx)):
                            if len(sub) < 2:
                                continue
                            dd = ma[key][sub] - mb[key][sub]
                            lo, hi = boot_ci(dd)
                            parts.append(f"d{key} {fm(nm(dd), lo, hi, 4 if key in ('nll_q', 'gain') else 3)}")
                            res[(T, d, lam, dn, key, a, b)] = (nm(dd), lo, hi)
                        log_fn(f"   {a}-{b}: " + " | ".join(parts))
                if "romantic_test" in ds and "positional_test" in ds:
                    for a, b in pairs:
                        for rn in ("romantic_test", "romantic_test_sem_XIX"):
                            ma, mb = M[(a, T, d, lam)], M[(b, T, d, lam)]
                            ir, ip = ds[rn], ds["positional_test"]
                            dr = ma["nll_q"][ir] - mb["nll_q"][ir]
                            dp = ma["nll_q"][ip] - mb["nll_q"][ip]
                            lo, hi = boot_diff_means(dr, dp)
                            val = nm(dr) - nm(dp)
                            log_fn(f"   >> INTERACAO {a}-{b} [{rn} menos positional_test] d(nll_q): "
                                   f"{fm(val, lo, hi)}   (tese: negativo para rom-pos)")
                            res[(T, d, lam, "interacao_" + rn, "nll_q", a, b)] = (val, lo, hi)
    if "rom" in names and "pos" in names:
        T0 = temps[0]
        if ref_lam in lams and ref_depth in depths:
            log_fn("")
            log_fn(f"=============== POR JOGADOR (T={T0}, d={ref_depth}, lambda={ref_lam}; d(nll_q)=rom-pos, "
                   f"negativo = romantica melhor) ===============")
            mr, mp = M[("rom", T0, ref_depth, ref_lam)], M[("pos", T0, ref_depth, ref_lam)]
            for dn in ("romantic_test", "positional_test"):
                if dn not in ds:
                    continue
                log_fn(f"[{dn}]")
                by = defaultdict(list)
                for i in ds[dn]:
                    by[trees[i].player].append(i)
                for p in sorted(by):
                    ii = np.array(by[p])
                    dd = mr["nll_q"][ii] - mp["nll_q"][ii]
                    dd = dd[~np.isnan(dd)]
                    if len(dd) < 2:
                        continue
                    lo, hi = boot_ci(dd)
                    era = " (sec. XIX)" if p.lower() in ERA19 else ""
                    log_fn(f"   {p:<13}{era:<12} n={len(dd):4d} d(nll_q) {fm(nm(dd), lo, hi)}")
    return res


def prune_stats(trees, pols, args, depths, lam, log_fn=log):
    name = next((n for n in pols if n not in ("rand", "random")), next(iter(pols)))
    pol = pols[name]
    sub = trees[:args.prune_n]
    log_fn("")
    log_fn(f"=============== EXPERIMENTO D: PODA (politica={name}, lambda={lam}, {len(sub)} posicoes) ===============")
    out = {}
    for d in depths:
        if d < 2:
            continue
        rows = {}
        bad = 0
        for label, prune, order in (("minimax puro", False, "policy"),
                                    ("alfa-beta, ordem pela politica", True, "policy"),
                                    ("alfa-beta, ordem pelo motor", True, "engine"),
                                    ("alfa-beta, ordem aleatoria", True, "random")):
            tot = np.zeros(3)
            vals = []
            for i, t in enumerate(sub):
                ctx = Ctx(lam=lam, pol=pol, T=args.temps[0], renorm=not args.no_renorm,
                          prune=prune, order=order, seed=i)
                Q = root_values(t.root, d, ctx)
                vals.append(max(Q))
                tot += np.array([ctx.n_own, ctx.n_opp, ctx.n_leaf])
            rows[label] = (tot / max(len(sub), 1), vals)
        base = rows["minimax puro"][1]
        for label, (_, vals) in rows.items():
            bad += int(sum(abs(a - b) > 1e-6 for a, b in zip(vals, base)))
        ref = rows["minimax puro"][0]
        log_fn(f"profundidade d={d}: nos proprios / nos do adversario / folhas / chamadas ao motor (proprios+adversario), "
               f"media por posicao")
        for label, (m, _) in rows.items():
            calls = m[0] + m[1]
            log_fn(f"   {label:<34} {m[0]:8.1f} {m[1]:8.1f} {m[2]:8.1f} | chamadas={calls:8.1f} "
                   f"({calls / (ref[0] + ref[1]):.0%} do minimax puro)")
        log_fn(f"   verificacao: valores da raiz diferentes do minimax puro em {bad} casos (esperado: 0)")
        out[d] = {k: v[0].tolist() for k, v in rows.items()}
        out[d]["mismatches"] = bad
    return out
