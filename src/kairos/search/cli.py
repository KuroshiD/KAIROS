r"""
kairos.search.cli   (KAIROS v2: busca adversarial hibrida)

FORMULACAO
  Stockfish  -> v(s): valor OBJETIVO de uma posicao, em centipawns, do ponto de vista do agente.
  Politica   -> pi(a|s): prior de ESTILO sobre lances proprios (nao e um valor de posicao).

  Utilidade de uma linha com lances proprios a_1..a_m e folha s_m:
        U = v(s_m) + lambda * sum_t log pi(a_t | s_t)        (soma so sobre lances PROPRIOS)
  lambda tem unidade de centipawns por nat (sem z-score).

  Decisao: minimax. Nos proprios maximizam U; nos do adversario minimizam U. O adversario nao tem
  estilo: ele so minimiza a qualidade objetiva. Profundidade d = numero de lances proprios.
      Q(a) = lambda*log pi(a|s) + min_{respostas b} V_own(s.a.b, d-1)        (d > 1)
      Q(a) = lambda*log pi(a|s) + cp(a)                                       (d = 1; cp = valor do Stockfish)
  Caso d = 1:  argmax_a  cp(a) + lambda*log pi(a|s)   e o modo de   pi(a) * exp(cp(a)/lambda)
  (otimo de  E_q[cp] - lambda*KL(q||pi) ). Decisao suave opcional: q(a) = softmax(Q(a)/lambda).

  Restricoes: so candidatos a ate --eps cp do melhor e sem mate contra (rede de seguranca).
  A arvore (candidatos e valores do motor) NAO depende de politica nem de lambda: e construida uma vez
  (modo build, banco SQLite) e resolvida offline para qualquer politica/lambda/temperatura (modo solve).

MODOS
  play   demonstracao em uma posicao (Stockfish ao vivo)
  build  consulta o Stockfish e grava a arvore das posicoes amostradas no banco (retomavel)
  solve  le o banco (sem Stockfish) e roda os experimentos A (trade-off), B (estilo), C (profundidade)
         e D (poda alfa-beta) com IC por bootstrap; grava logs, JSON e CSV
"""
import argparse
import csv
import json
import os
import platform
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import chess
import numpy as np

from kairos.engine import Analyzer
from kairos.search.common import log, softmax
from kairos.search.evaluation import METRICS, nm, prune_stats, run_configs, sample_positions, summarize
from kairos.search.minimax import Ctx, argbest, root_values
from kairos.search.policy import node_logpi, parse_policies, score_all
from kairos.search.tree import (
    Provider,
    TreeDB,
    collect_need,
    load_trees,
    make_own,
    tree_params,
)


def add_common(ap):
    ap.add_argument("--k-own", type=int, default=6, help="largura nos nos proprios")
    ap.add_argument("--k-opp", type=int, default=3, help="largura nos nos do adversario")
    ap.add_argument("--eps", type=float, default=100, help="janela (cp) dos candidatos proprios")
    ap.add_argument("--opp-eps", type=float, default=150, help="janela (cp) das respostas do adversario")
    ap.add_argument("--nodes", type=int, default=5000, help="nos do Stockfish na raiz")
    ap.add_argument("--root-k", type=int, default=16, help="MultiPV da raiz (depois filtra a janela e k-own)")
    ap.add_argument("--leaf-nodes", type=int, default=1000, help="nos do Stockfish nos nos proprios internos")
    ap.add_argument("--opp-nodes", type=int, default=1000, help="nos do Stockfish nos nos do adversario")
    ap.add_argument("--no-renorm", action="store_true",
                    help="nao renormaliza pi dentro dos candidatos (usa todos os lances legais)")
    ap.add_argument("--seed", type=int, default=1)


def add_positions(ap):
    ap.add_argument("--romantic", default=os.path.join("data", "style-data", "style_romantic.jsonl"))
    ap.add_argument("--positional", default=os.path.join("data", "style-data", "style_positional.jsonl"))
    ap.add_argument("--fischer", default=os.path.join("data", "style-data", "style_test.jsonl"))
    ap.add_argument("--n", type=int, default=1200, help="posicoes por conjunto")
    ap.add_argument("--no-balance", action="store_true")


def cmd_build(args):
    P = args
    db = TreeDB(args.db)
    pos = db.positions()
    if pos:
        log(f"banco ja tem {len(pos)} posicoes; reutilizando (o amostrador nao sera executado)")
    else:
        recs = sample_positions(args)
        db.add_positions(recs)
        pos = db.positions()
        log(f"{len(pos)} posicoes gravadas no banco")
    old = db.get_meta("tree_params")
    if old and old != tree_params(args):
        log(f"AVISO: parametros diferem dos do banco. banco={old} agora={tree_params(args)}")
    db.set_meta("tree_params", tree_params(args))
    db.set_meta("depth_built", args.depth)
    seeds = {}
    if args.seed_json and os.path.exists(args.seed_json):
        raw = json.load(open(args.seed_json, encoding="utf-8"))
        seeds = {k: [[c["uci"], c["cp"], c["mate"]] for c in v] for k, v in raw.items()}
        log(f"cache semente carregado: {len(seeds)} analises ({args.seed_json})")
    log(f"banco: {args.db} | analises ja gravadas: {db.n_analysis()}")
    an = Analyzer(args.stockfish, args.engines)
    prov = Provider(db, an, seeds)
    errors, done, t0 = [], [0], time.time()
    lock = threading.Lock()

    def work(row):
        pid, dataset, player, fen, human = row
        try:
            make_own(prov, fen, args.depth, P, root=True)
        except Exception:
            errors.append((fen, traceback.format_exc(limit=3)))
        with lock:
            done[0] += 1
            if done[0] % 50 == 0:
                log(f"  posicoes: {done[0]}/{len(pos)} | motor={prov.stats['sf']} banco={prov.stats['db']} "
                    f"semente={prov.stats['seed']} | {time.time() - t0:.0f}s | erros={len(errors)}")

    try:
        with ThreadPoolExecutor(args.engines) as ex:
            list(ex.map(work, pos))
    finally:
        db.commit()
        an.close()
    log(f"build concluido em {time.time() - t0:.0f}s | chamadas ao motor={prov.stats['sf']} | "
        f"analises no banco={db.n_analysis()} | erros={len(errors)}")
    for fen, tb in errors[:5]:
        log(f"  ERRO fen={fen}\n{tb}")
    log("FIM")


def cmd_solve(args):
    import torch
    log(f"python {platform.python_version()} | torch {torch.__version__} | cuda={torch.cuda.is_available()}")
    log("argumentos: " + json.dumps(vars(args), ensure_ascii=False))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    db = TreeDB(args.db)
    meta = db.get_meta("tree_params")
    log(f"parametros da arvore no banco: {meta} | profundidade construida: {db.get_meta('depth_built')}")
    if meta and meta != tree_params(args):
        log(f"AVISO: parametros de arvore diferentes dos do build. banco={meta} agora={tree_params(args)}")
    depths = [int(x) for x in args.depths.split(",")]
    lams = [float(x) for x in args.lams.split(",")]
    args.temps = [float(x) for x in args.temps.split(",")]
    trees = load_trees(db, args, max(depths))
    if not trees:
        log("NENHUMA arvore completa; nada a resolver. Envie este log.")
        return
    pols_raw = parse_policies(args.policy, device)
    scored = score_all(trees, pols_raw, args.temps)
    pols = {n: s for n, s in scored.items()}
    names = list(pols)
    cov = np.mean([any(c.uci == t.human for c in t.root.cands) for t in trees])
    log(f"lance humano entre os candidatos da raiz (k-own={args.k_own}, eps={args.eps} cp): {cov:.1%}")
    t0 = time.time()
    M = run_configs(trees, pols, args, depths, lams, args.temps)
    log(f"configuracoes resolvidas em {time.time() - t0:.0f}s")
    ref_depth = args.ref_depth or max(depths)
    res = summarize(trees, M, names, depths, lams, args.temps, args.ref_lam, ref_depth)
    pr = prune_stats(trees, pols, args, depths, args.ref_lam) if args.prune_n > 0 else {}
    if args.out:
        out = {"args": {k: v for k, v in vars(args).items()}, "n_trees": len(trees),
               "results": {"|".join(map(str, k)): {"mean": v[0], "lo": v[1], "hi": v[2]} for k, v in res.items()},
               "means": {"|".join(map(str, k)): {m: nm(arr[m]) for m in METRICS} for k, arr in M.items()},
               "prune": pr}
        json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2, default=str)
        log(f"resultados -> {args.out}")
    if args.dump:
        with open(args.dump, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "player", "fen", "human", "policy", "T", "depth", "lam"] + list(METRICS))
            for (name, T, d, lam), arr in M.items():
                if lam != args.ref_lam:
                    continue
                for i, t in enumerate(trees):
                    w.writerow([t.dataset, t.player, t.fen, t.human, name, T, d, lam]
                               + [arr[m][i] for m in METRICS])
        log(f"dump (lambda={args.ref_lam}) -> {args.dump}")
    log("FIM")


def cmd_play(args):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.temps = [float(x) for x in args.temps.split(",")]
    pols_raw = parse_policies(args.policy, device)
    db = TreeDB(args.db or ":memory:")
    an = Analyzer(args.stockfish, args.engines)
    try:
        prov = Provider(db, an)
        root = make_own(prov, args.fen, args.depth, args, root=True)
        log(f"arvore construida: chamadas ao motor={prov.stats['sf']}")
    finally:
        an.close()
    need = {}
    collect_need(root, need)
    b = chess.Board(args.fen)
    T = args.temps[0]
    q0 = root_values(root, args.depth, Ctx(lam=0.0))
    for name, p in pols_raw.items():
        pol = p.raw(need, args.temps)
        ctx = Ctx(lam=args.lam, pol=pol, T=T, renorm=not args.no_renorm)
        Q = root_values(root, args.depth, ctx)
        lp = node_logpi(root, pol, T, not args.no_renorm)
        j = argbest(Q, root.cands)
        a0 = argbest(q0, root.cands)
        print(f"\n[{name}] lambda={args.lam} cp/nat, profundidade={args.depth}: joga {b.san(chess.Move.from_uci(root.cands[j].uci))} "
              f"(objetivo puro: {b.san(chess.Move.from_uci(root.cands[a0].uci))}; "
              f"perda objetiva {max(q0) - q0[j]:.0f} cp)")
        print("  lance     cp_raiz   log pi    lambda*logpi   Q(a)   q_suave")
        qs = softmax(np.array(Q) / args.lam) if args.lam > 0 else np.zeros(len(Q))
        for i in sorted(range(len(Q)), key=lambda k: -Q[k]):
            c = root.cands[i]
            print(f"  {b.san(chess.Move.from_uci(c.uci)):<8}{c.cp:8.0f}{lp[i]:10.2f}{args.lam * lp[i]:13.1f}{Q[i]:9.1f}{qs[i]:9.3f}")
        for label, prune, order in (("minimax puro", False, "policy"), ("alfa-beta (ordem pela politica)", True, "policy")):
            cx = Ctx(lam=args.lam, pol=pol, T=T, renorm=not args.no_renorm, prune=prune, order=order)
            Qp = root_values(root, args.depth, cx)
            print(f"  {label}: nos proprios={cx.n_own} adversario={cx.n_opp} folhas={cx.n_leaf} "
                  f"melhor valor={max(Qp):.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play")
    add_common(p)
    p.add_argument("--stockfish", required=True)
    p.add_argument("--policy", action="append", required=True)
    p.add_argument("--fen", required=True)
    p.add_argument("--lam", type=float, default=20.0)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--temps", default="1.0")
    p.add_argument("--db", default=None)
    p.add_argument("--engines", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--cpu", action="store_true")

    p = sub.add_parser("build")
    add_common(p)
    add_positions(p)
    p.add_argument("--stockfish", required=True)
    p.add_argument("--db", default=os.path.join("runs", "kairos_tree.sqlite"))
    p.add_argument("--depth", type=int, default=2, help="lances proprios a construir")
    p.add_argument("--seed-json", default=None, help="cache JSON da decisao cruzada (reaproveita a raiz)")
    p.add_argument("--engines", type=int, default=max(1, (os.cpu_count() or 2) - 1))

    p = sub.add_parser("solve")
    add_common(p)
    p.add_argument("--db", default=os.path.join("runs", "kairos_tree.sqlite"))
    p.add_argument("--policy", action="append", required=True, help="nome=caminho.pt ou nome=random")
    p.add_argument("--depths", default="1,2")
    p.add_argument("--lams", default="0,5,10,20,40,80,160")
    p.add_argument("--temps", default="1.0")
    p.add_argument("--ref-lam", type=float, default=20.0)
    p.add_argument("--ref-depth", type=int, default=0)
    p.add_argument("--prune-n", type=int, default=300, help="posicoes do experimento D (0 = desliga)")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", default=os.path.join("runs", "v2_results.json"))
    p.add_argument("--dump", default=os.path.join("runs", "v2_positions.csv"))

    args = ap.parse_args()
    {"play": cmd_play, "build": cmd_build, "solve": cmd_solve}[args.cmd](args)


if __name__ == "__main__":
    main()
