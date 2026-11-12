"""
kairos.models.style_finetune  (M4: ajuste fino de estilo)

Fine-tuning de estilo com warm-start a partir da policy do 02:

  --group romantic    Morphy + Tal + Kasparov   (Anderssen so como referencia)
  --group positional  Karpov + Petrosian + Capablanca   (politica de CONTROLE, H2)

Rode os dois com os MESMOS hiperparametros e o mesmo --max-positions, para que
a diferenca entre eles seja atribuivel ao corpus e nao ao treino.

Por epoca:
  1. treina em lotes: CE nos lances do jogador-alvo (ponderada) + REPLAY do
     dataset de forca (loss completa do 02) + KL da policy original no replay;
  2. mede no VAL de estilo (por partida, sem vazamento): NLL, top-1 e o desvio
     das taxas de captura/xeque/sacrificio/... entre o lance previsto e o humano;
  3. gate de nao-regressao tatica (top-1 de PGN e puzzle no val do 02);
  4. adapta os pesos por flag: se o modelo faz menos X que o jogador, as
     posicoes em que o jogador fez X pesam mais na epoca seguinte;
  5. guarda best.pt = menor NLL de estilo entre as epocas que passaram no gate.

O TEST de estilo so e usado no fim.

Uso:
  kairos-corpus build ...  ->  data/style-data/style_romantic.jsonl
  kairos-finetune --policy runs/p1/best.pt --data data_policy \
      --style-data data/style-data --group romantic --out runs/style_rom
  kairos-finetune ... --group positional --out runs/style_pos
"""

import argparse
import json
import math
import os
import time
from types import SimpleNamespace

import chess
import numpy as np
import torch
import torch.nn as nn

from kairos.chess_ext import (
    ACTION_SIZE,
    SplitData,
    board_to_planes,
    encode_move,
    legal_action_ids,
)
from kairos.data.style_corpus import FLAGS, load_jsonl, move_flags

from kairos.models import train as tp  # noqa: E402

LOSS_ARGS = SimpleNamespace(
    label_smoothing=0.03, puzzle_weight=1.0,
    w_from=0.2, w_to=0.2, w_phase=0.1,
    w_mtype=0.2, w_legal=0.2, w_theme=0.1,
)


# ============================================================
# DADOS
# ============================================================

def prep(rows, args, rng):
    """Planos, alvo, mascara de legais, flags e peso base de cada posicao."""

    D = {k: [] for k in ("fen", "X", "Y", "L", "F", "W")}
    dropped = 0

    for r in rows:

        # descarta lances humanos muito ruins (adversarios fracos do seculo XIX etc.)
        if args.drop_blunders and r.get("cp_loss", 0) > args.drop_blunders:
            dropped += 1
            continue

        b = chess.Board(r["fen"])
        m = chess.Move.from_uci(r["uci"])
        ids = np.fromiter((int(a) for a in legal_action_ids(b)), np.int32)
        a = int(encode_move(m))

        if a not in set(ids.tolist()):
            dropped += 1
            continue

        # o estilo mora onde o humano NAO jogou o top-1 do Stockfish
        w = 1.0 + (args.diverge_bonus if r.get("rank", 0) != 0 else 0.0)

        D["fen"].append(r["fen"])
        D["X"].append(board_to_planes(b).astype(np.float16))
        D["Y"].append(a)
        D["L"].append(ids)
        D["F"].append(move_flags(b, m))
        D["W"].append(w)

    for k in ("X", "F", "W"):
        if k == "W":
            D[k] = np.asarray(D[k], np.float32)
        else:
            D[k] = np.stack(D[k]) if D[k] else np.empty((0,))

    D["Y"] = np.asarray(D["Y"], np.int64)

    if args.max_positions and len(D["Y"]) > args.max_positions:
        keep = np.sort(rng.choice(len(D["Y"]), args.max_positions, replace=False))
        for k in ("X", "F", "W", "Y"):
            D[k] = D[k][keep]
        D["fen"] = [D["fen"][i] for i in keep]
        D["L"] = [D["L"][i] for i in keep]

    D["dropped"] = dropped
    return D


def make_batch(D, idx, device):

    x = torch.from_numpy(D["X"][idx]).float()
    y = torch.from_numpy(D["Y"][idx]).long()
    leg = torch.zeros(len(idx), ACTION_SIZE, dtype=torch.bool)

    for j, i in enumerate(idx):
        leg[j, torch.from_numpy(D["L"][i]).long()] = True

    return x.to(device), y.to(device), leg.to(device)


def replay_stream(ds, workers, pin):

    epoch = 0

    while True:
        ds.epoch = epoch
        epoch += 1
        for batch in tp.make_loader(ds, workers, pin):
            yield batch


# ============================================================
# AVALIACAO
# ============================================================

@torch.no_grad()
def style_eval(ctx, model, D, bs=512):

    model.eval()
    n = len(D["Y"])
    nll = top1 = 0.0

    for s in range(0, n, bs):

        idx = np.arange(s, min(n, s + bs))
        x, y, leg = make_batch(D, idx, ctx.device)

        with torch.autocast(ctx.device.type, dtype=ctx.amp_dtype,
                            enabled=ctx.amp_dtype is not None):
            logits = model(x)["policy"]

        logits = logits.float().masked_fill(~leg, -1e9)
        lp = torch.log_softmax(logits, 1)

        nll += float(-lp.gather(1, y[:, None]).sum())
        top1 += float((logits.argmax(1) == y).sum())

    return nll / n, top1 / n


@torch.no_grad()
def model_flag_rates(ctx, model, D, idx, bs=256):
    """Taxa de cada flag nos lances ESCOLHIDOS (argmax) pelo modelo."""

    model.eval()
    acc = np.zeros(len(FLAGS), np.float64)

    for s in range(0, len(idx), bs):

        sub = idx[s:s + bs]
        x, y, leg = make_batch(D, sub, ctx.device)

        with torch.autocast(ctx.device.type, dtype=ctx.amp_dtype,
                            enabled=ctx.amp_dtype is not None):
            logits = model(x)["policy"]

        pred = logits.float().masked_fill(~leg, -1e9).argmax(1).cpu().numpy()

        for j, i in enumerate(sub):
            b = chess.Board(D["fen"][i])
            mp = dict(zip((int(a) for a in legal_action_ids(b)), b.legal_moves))
            mv = mp.get(int(pred[j]))
            if mv is not None:
                acc += move_flags(b, mv)

    return acc / max(1, len(idx))


def tactical_metrics(ctx, model):
    loader = tp.make_loader(ctx.val_ds, ctx.args.workers, ctx.pin)
    return tp.evaluate(model, loader, ctx.device, ctx.amp_dtype)


def regression_ok(ctx, res):

    ok, msgs = True, []

    for k in ("pgn", "puzzle"):
        if k in ctx.base and k in res:
            d = res[k]["top1"] - ctx.base[k]["top1"]
            msgs.append(f"{k} {res[k]['top1']:.4f} ({d:+.4f})")
            if -d > ctx.args.max_drop:
                ok = False

    return ok, msgs


def save_ckpt(ctx, path, epoch, extra=None):
    torch.save({"model": ctx.ema.shadow.state_dict(),
                "raw_model": ctx.model.state_dict(),
                "config": ctx.cfg, "group": ctx.args.group,
                "epoch": epoch, **(extra or {})}, path)


# ============================================================
# MAIN
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("--policy", required=True, help="best.pt do 02")
    p.add_argument("--data", required=True, help="pasta do 01 (replay + val)")
    p.add_argument("--style-file", default=None,
                   help="jsonl de estilo direto")
    p.add_argument("--style-data", default=None,
                   help="pasta com style_<grupo>.jsonl")
    p.add_argument("--group", default=None,
                   help="nome do corpus existente em --style-data")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5,
                   help="baixo: mitiga esquecimento tatico (PDF 5.1)")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--style-batch", type=int, default=128)
    p.add_argument("--replay-batch", type=int, default=256)
    p.add_argument("--replay-weight", type=float, default=1.0)
    p.add_argument("--kl-weight", type=float, default=0.5,
                   help="KL da policy original, calculada so no replay")
    p.add_argument("--label-smoothing", type=float, default=0.03)
    p.add_argument("--ema", type=float, default=0.995)
    p.add_argument("--max-drop", type=float, default=0.01)
    p.add_argument("--eval-max", type=int, default=50000)

    p.add_argument("--max-positions", type=int, default=0,
                   help="iguala o tamanho do treino entre romantic e positional")
    p.add_argument("--drop-blunders", type=int, default=2500,
                   help="descarta lances humanos que perdem > N cp (0 = nao)")
    p.add_argument("--diverge-bonus", type=float, default=1.0,
                   help="peso extra onde o humano nao jogou o top-1 do SF")
    p.add_argument("--bonus-lr", type=float, default=0.5,
                   help="passo da adaptacao por flag entre epocas (0 = desliga)")
    p.add_argument("--adapt-sample", type=int, default=4000)

    args = p.parse_args()

    if not args.style_file:
        if not (args.style_data and args.group):
            p.error("informe --style-file OU (--style-data e --group)")
        args.style_file = os.path.join(args.style_data,
                                       f"style_{args.group}.jsonl")

    if not args.group:
        stem = os.path.splitext(os.path.basename(args.style_file))[0]
        args.group = stem[6:] if stem.startswith("style_") else stem

    return args


def main():

    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp_dtype = torch.bfloat16 if (
        device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
    pin = device.type == "cuda"

    print(f"device={device} | amp={amp_dtype or 'off'} | grupo={args.group}")

    # ---- corpus de estilo (split por PARTIDA, feito no build)
    path = args.style_file
    tr_rows = load_jsonl(path, {"train"})
    va_rows = load_jsonl(path, {"val"})
    te_rows = load_jsonl(path, {"test"})

    tr = prep(tr_rows, args, rng)
    va = prep(va_rows, args, rng)
    te = prep(te_rows, args, rng)

    print(f"estilo: treino {len(tr['Y']):,} | val {len(va['Y']):,} | "
          f"test {len(te['Y']):,} (descartadas {tr['dropped']:,} no treino)")

    if len(va["Y"]) == 0:
        raise SystemExit("val de estilo vazio: poucas partidas no PGN "
                         "(o split e por partida, ~10% val / 10% test)")

    if len(tr["Y"]) < args.style_batch:
        raise SystemExit("corpus de estilo pequeno demais")

    # ---- replay + val tatico
    train = SplitData(os.path.join(args.data, "train"))
    val = SplitData(os.path.join(args.data, "val"))

    pool_idx = None
    if val.n > args.eval_max:
        pool_idx = np.sort(np.random.default_rng(args.seed).choice(
            val.n, args.eval_max, replace=False))

    val_ds = tp.BatchDataset(val, 512, False, 0, pool=pool_idx)
    replay_ds = tp.BatchDataset(train, args.replay_batch, True, args.seed,
                                drop_last=True)
    replay_iter = replay_stream(replay_ds, args.workers, pin)

    # ---- modelos
    ck = torch.load(args.policy, map_location=device, weights_only=False)
    cfg = ck["config"]

    teacher = tp.PolicyNet(**cfg).to(device)
    teacher.load_state_dict(ck["model"])
    teacher.eval()
    for q in teacher.parameters():
        q.requires_grad_(False)

    model = tp.PolicyNet(**cfg).to(device)
    model.load_state_dict(ck["model"])
    ema = tp.EMA(model, args.ema)

    ctx = SimpleNamespace(args=args, device=device, amp_dtype=amp_dtype,
                          pin=pin, cfg=cfg, model=model, ema=ema,
                          val_ds=val_ds, base=None)

    ctx.base = tactical_metrics(ctx, teacher)
    tp.print_metrics("VAL tatico da policy original", ctx.base)

    n_tr = len(tr["Y"])
    sample = rng.choice(n_tr, min(args.adapt_sample, n_tr), replace=False)
    human_rate = tr["F"][sample].mean(0)

    base_nll, base_top1 = style_eval(ctx, teacher, va)
    base_rate = model_flag_rates(ctx, teacher, tr, sample)

    print(f"estilo VAL da policy original: nll={base_nll:.4f} "
          f"top1={base_top1:.4f}")
    print("  flags (humano | original): " + " ".join(
        f"{f}={h:.3f}|{m:.3f}" for f, h, m in zip(FLAGS, human_rate, base_rate)))

    # ---- otimizador
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]

    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.02},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.98))

    steps_ep = math.ceil(n_tr / args.style_batch)
    total = steps_ep * args.epochs
    bonus = np.zeros(len(FLAGS), np.float32)
    mult = np.ones(n_tr, np.float32)

    best_nll, bad, gstep = float("inf"), 0, 0

    for ep in range(1, args.epochs + 1):

        model.train()
        perm = rng.permutation(n_tr)
        run = {}
        t0 = time.time()

        for s in range(steps_ep):

            lr = args.lr * min(1.0, (gstep + 1) / args.warmup) * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * gstep / total)))
            for g in opt.param_groups:
                g["lr"] = lr

            idx = perm[s * args.style_batch:(s + 1) * args.style_batch]

            if len(idx) == 0:
                continue

            xs, ys, ls = make_batch(tr, idx, device)
            ws = torch.from_numpy(tr["W"][idx] * mult[idx]).to(device)

            rb = tp.to_device_batch(next(replay_iter), device)
            x = torch.cat([xs, rb["x"]])

            with torch.autocast(device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                out = model(x)

            B = xs.shape[0]
            out_s = {k: v[:B] for k, v in out.items()}
            out_r = {k: v[B:] for k, v in out.items()}

            ce = tp.masked_ce(out_s["policy"], ls, ys, args.label_smoothing)
            l_s = (ce * ws).sum() / ws.sum()
            l_r, _ = tp.compute_loss(out_r, rb, LOSS_ARGS)

            loss = l_s + args.replay_weight * l_r
            kl_v = 0.0

            if args.kl_weight > 0:
                with torch.no_grad(), torch.autocast(
                        device.type, dtype=amp_dtype,
                        enabled=amp_dtype is not None):
                    t_pol = teacher(rb["x"])["policy"]

                lp_s = tp.masked_logp(out_r["policy"], rb["legal"])
                lp_t = tp.masked_logp(t_pol, rb["legal"])
                kl = (lp_t.exp() * (lp_t - lp_s)).sum(1).mean()
                loss = loss + args.kl_weight * kl
                kl_v = float(kl.item())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model, gstep)
            gstep += 1

            for k, v in (("estilo", float(l_s.item())),
                         ("replay", float(l_r.item())), ("kl", kl_v)):
                run[k] = run.get(k, v) * 0.98 + v * 0.02

        # ---- fim da epoca
        nll, top1 = style_eval(ctx, ema.shadow, va)
        rate = model_flag_rates(ctx, ema.shadow, tr, sample)
        res = tactical_metrics(ctx, ema.shadow)
        ok, msgs = regression_ok(ctx, res)

        gap = human_rate - rate

        print(f"epoca {ep}/{args.epochs} ({(time.time() - t0) / 60:.1f} min) | "
              + " ".join(f"{k}={v:.3f}" for k, v in run.items())
              + f" | VAL estilo nll={nll:.4f} top1={top1:.4f}"
              + f" | tatico: {' '.join(msgs)}"
              + (" -> ok" if ok else " -> REGRESSAO"), flush=True)
        print("  flag (humano-modelo): " + " ".join(
            f"{f}={g:+.3f}" for f, g in zip(FLAGS, gap)), flush=True)

        with open(os.path.join(args.out, "log.jsonl"), "a") as f:
            f.write(json.dumps({
                "epoch": ep, "style_nll": nll, "style_top1": top1,
                "flag_gap": dict(zip(FLAGS, map(float, gap))),
                "tactical": {k: res[k] for k in ("all", "pgn", "puzzle")
                             if k in res},
                "regression_ok": ok, "time": time.time()}) + "\n")

        if ok and nll < best_nll - 1e-4:
            best_nll, bad = nll, 0
            save_ckpt(ctx, os.path.join(args.out, "best.pt"), ep,
                      {"style_nll": nll})
            print("  * novo best.pt", flush=True)
        else:
            bad += 1
            if bad >= args.patience:
                print("  parada antecipada")
                break

        # ---- adaptacao por flag: onde o modelo faz MENOS do que o jogador,
        # sobe o peso das posicoes em que o jogador fez aquilo
        if args.bonus_lr > 0:
            rel = gap / (human_rate + 0.05)
            bonus = np.clip(bonus + args.bonus_lr * rel, -1.0, 1.0)
            mult = np.exp(tr["F"] @ bonus).astype(np.float32)
            mult = np.clip(mult, 0.5, 3.0)
            print("  bonus por flag: " + " ".join(
                f"{f}={b:+.2f}" for f, b in zip(FLAGS, bonus)), flush=True)

    # ---- relatorio final no TEST de estilo
    if os.path.exists(os.path.join(args.out, "best.pt")) and len(te["Y"]):

        bk = torch.load(os.path.join(args.out, "best.pt"),
                        map_location=device, weights_only=False)
        final = tp.PolicyNet(**cfg).to(device)
        final.load_state_dict(bk["model"])

        t_nll, t_top1 = style_eval(ctx, final, te)
        o_nll, o_top1 = style_eval(ctx, teacher, te)

        print(f"\nTEST estilo ({args.group}): nll {o_nll:.4f} -> {t_nll:.4f} | "
              f"top1 {o_top1:.4f} -> {t_top1:.4f}")

        with open(os.path.join(args.out, "test_style.json"), "w") as f:
            json.dump({"group": args.group,
                       "original": {"nll": o_nll, "top1": o_top1},
                       "finetuned": {"nll": t_nll, "top1": t_top1}}, f,
                      indent=2)


if __name__ == "__main__":
    main()