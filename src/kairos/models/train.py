"""
kairos.models.train

Treino de policy PURA (sem value/MCTS) sobre o dataset de
01_prepare_policy_dataset.py (PGN + puzzles).

Nada de parametro morto:
  * cabeca de policy com 1.968 saidas 
  * as cabecas from/to NAO sao penduradas a toa: os logits delas SOMAM
    no logit de cada lance (logit = fc + from[f] + to[t]), alem de
    terem loss auxiliar propria;
  * os 4 bits mortos de `tags` viraram alvo auxiliar (castle,
    enPassant, promotion, capture do lance alvo);
  * cabecas auxiliares extras que forcam o tronco a aprender REGRAS:
    casas de origem/destino legais (pins, roque, en passant, xeque),
    fase da partida e temas de puzzle;
  * no 1o passo o script confere que TODO parametro recebeu gradiente.

Loss = CE mascarada pelos lances legais (label smoothing so entre os
legais) + auxiliares ponderadas. Avaliacao: top-1/top-3 e loss por
fase, por fonte (pgn/puzzle) e por tipo de lance (castle, en passant,
promocao, captura). EMA dos pesos e usada para avaliar e salvar.

Uso:
  kairos-train --data data/data_policy --out runs/p1 \
      --epochs 10 --batch-size 512 --workers 4
"""

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from kairos.chess_ext import (
    ACTION_FROM,
    ACTION_SIZE,
    ACTION_TO,
    INPUT_CHANNELS,
    MOVE_TAG_NAMES,
    PHASE_NAMES,
    PUZZLE_GROUP_OFFSET,
    TAG_BIT,
    THEME_TAG_NAMES,
    SplitData,
)
from kairos.models.policy_net import PolicyNet, load_policy



# ============================================================
# DATA
# ============================================================

class BatchDataset(Dataset):
    """Cada item e um BATCH inteiro (use DataLoader(batch_size=None))."""

    def __init__(self, split, batch_size, shuffle, seed, pool=None,
                 drop_last=False):

        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.pool = pool
        self.drop_last = drop_last
        self.epoch = 0
        self._order_cache = (None, None)

        self.n = split.n if pool is None else len(pool)

    def __len__(self):

        if self.drop_last:
            return self.n // self.batch_size

        return math.ceil(self.n / self.batch_size)

    def _order(self):

        if self._order_cache[0] != self.epoch:

            if self.shuffle:
                rng = np.random.default_rng(self.seed + 7919 * self.epoch)
                order = rng.permutation(self.n)
            else:
                order = np.arange(self.n)

            if self.pool is not None:
                order = self.pool[order]

            self._order_cache = (self.epoch, order)

        return self._order_cache[1]

    def __getitem__(self, i):

        idx = self._order()[i * self.batch_size:(i + 1) * self.batch_size]

        return self.split.gather(np.sort(idx))


def make_loader(dataset, workers, pin):

    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        prefetch_factor=4 if workers > 0 else None,
        persistent_workers=False,
    )


_SHIFTS = torch.arange(7, -1, -1, dtype=torch.int16)

_TAG_BITS = torch.tensor(
    [TAG_BIT[t] for t in PHASE_NAMES], dtype=torch.int32
)
_THEME_BITS = torch.tensor(
    [TAG_BIT[t] for t in THEME_TAG_NAMES], dtype=torch.int32
)
_MOVE_BITS = torch.tensor(
    [TAG_BIT[t] for t in MOVE_TAG_NAMES], dtype=torch.int32
)


def to_device_batch(batch, device):
    """CPU batch (CSR) -> tensores de treino na GPU."""

    X = batch["X"].to(device, non_blocking=True).to(torch.int16)

    B = X.shape[0]

    bits = (X.unsqueeze(-1) >> _SHIFTS.to(device)) & 1
    x = bits.reshape(B, INPUT_CHANNELS, 8, 8).float()

    y = batch["y"].to(device, non_blocking=True).long()
    tags = batch["tags"].to(device, non_blocking=True)
    group = batch["group"].to(device, non_blocking=True)

    offsets = batch["legal_offsets"].to(device, non_blocking=True)
    flat = batch["legal_flat"].to(device, non_blocking=True).long()

    lengths = offsets[1:] - offsets[:-1]
    rows = torch.repeat_interleave(torch.arange(B, device=device), lengths)

    a_from = torch.as_tensor(ACTION_FROM, device=device)
    a_to = torch.as_tensor(ACTION_TO, device=device)

    legal = torch.zeros(B, ACTION_SIZE, dtype=torch.bool, device=device)
    legal[rows, flat] = True

    legal_from = torch.zeros(B, 64, dtype=torch.bool, device=device)
    legal_to = torch.zeros(B, 64, dtype=torch.bool, device=device)
    legal_from[rows, a_from[flat]] = True
    legal_to[rows, a_to[flat]] = True

    phase_hit = (tags.unsqueeze(1) & _TAG_BITS.to(device)) != 0
    phase = phase_hit.float().argmax(1)

    themes = ((tags.unsqueeze(1) & _THEME_BITS.to(device)) != 0).float()
    mtypes = ((tags.unsqueeze(1) & _MOVE_BITS.to(device)) != 0).float()

    return {
        "x": x,
        "y": y,
        "legal": legal,
        "legal_from": legal_from,
        "legal_to": legal_to,
        "from_t": a_from[y],
        "to_t": a_to[y],
        "phase": phase,
        "themes": themes,
        "mtypes": mtypes,
        "is_puzzle": group >= PUZZLE_GROUP_OFFSET,
    }


# ============================================================
# LOSS
# ============================================================

def masked_logp(logits, mask):
    return F.log_softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)


def masked_ce(logits, mask, target, eps=0.0):
    """CE por linha sobre os lances legais; suavizacao so entre legais."""

    logp = masked_logp(logits, mask)

    nll = -logp.gather(1, target[:, None]).squeeze(1)

    if eps <= 0:
        return nll

    n = mask.sum(1).clamp(min=1)

    smooth = -(logp * mask.float()).sum(1) / n

    return (1 - eps) * nll + eps * smooth


def compute_loss(out, b, args):

    w = torch.where(
        b["is_puzzle"],
        torch.full_like(b["y"], args.puzzle_weight, dtype=torch.float32),
        torch.ones_like(b["y"], dtype=torch.float32),
    )

    def wmean(v):
        return (v * w).sum() / w.sum()

    l_pol = masked_ce(out["policy"], b["legal"], b["y"], args.label_smoothing)
    l_from = masked_ce(out["from"], b["legal_from"], b["from_t"])
    l_to = masked_ce(out["to"], b["legal_to"], b["to_t"])

    l_phase = F.cross_entropy(out["phase"].float(), b["phase"], reduction="none")

    l_mtype = F.binary_cross_entropy_with_logits(
        out["mtype"].float(), b["mtypes"], reduction="none"
    ).mean(1)

    ls = out["legal_sq"].float()

    l_legal = 0.5 * (
        F.binary_cross_entropy_with_logits(
            ls[:, 0], b["legal_from"].float(), reduction="none").mean(1)
        + F.binary_cross_entropy_with_logits(
            ls[:, 1], b["legal_to"].float(), reduction="none").mean(1)
    )

    # temas: so linhas de puzzle
    pz = b["is_puzzle"].float()

    l_theme = F.binary_cross_entropy_with_logits(
        out["theme"].float(), b["themes"], reduction="none"
    ).mean(1)

    l_theme = (l_theme * pz).sum() / pz.sum().clamp(min=1)

    total = (
        wmean(l_pol)
        + args.w_from * wmean(l_from)
        + args.w_to * wmean(l_to)
        + args.w_phase * wmean(l_phase)
        + args.w_mtype * wmean(l_mtype)
        + args.w_legal * wmean(l_legal)
        + args.w_theme * l_theme
    )

    parts = {
        "policy": l_pol.mean().item(),
        "from": l_from.mean().item(),
        "to": l_to.mean().item(),
        "phase": l_phase.mean().item(),
        "mtype": l_mtype.mean().item(),
        "legal": l_legal.mean().item(),
        "theme": float(l_theme.item()),
    }

    return total, parts


# ============================================================
# EMA
# ============================================================

class EMA:

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):

        d = min(self.decay, (1 + step) / (10 + step))

        for s, m in zip(self.shadow.state_dict().values(),
                        model.state_dict().values()):

            if s.dtype.is_floating_point:
                s.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                s.copy_(m)


# ============================================================
# EVALUATION
# ============================================================

class Meter:

    def __init__(self):
        self.d = {}

    def add(self, name, mask, hit1, hit3, nll):

        if not mask.any():
            return

        c = self.d.setdefault(name, [0, 0.0, 0.0, 0.0])

        c[0] += int(mask.sum())
        c[1] += float(hit1[mask].sum())
        c[2] += float(hit3[mask].sum())
        c[3] += float(nll[mask].sum())

    def result(self):
        return {
            k: {
                "n": v[0],
                "top1": v[1] / v[0],
                "top3": v[2] / v[0],
                "nll": v[3] / v[0],
            }
            for k, v in self.d.items()
        }


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):

    model.eval()

    meter = Meter()

    for batch in loader:

        b = to_device_batch(batch, device)

        with torch.autocast(device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            out = model(b["x"])

        logits = out["policy"].float().masked_fill(~b["legal"], -1e9)

        top3 = logits.topk(3, dim=1).indices

        hit1 = (top3[:, 0] == b["y"]).float()
        hit3 = (top3 == b["y"][:, None]).any(1).float()

        nll = -F.log_softmax(logits, -1).gather(1, b["y"][:, None]).squeeze(1)

        allm = torch.ones_like(hit1, dtype=torch.bool)

        meter.add("all", allm, hit1, hit3, nll)

        src = {"puzzle": b["is_puzzle"], "pgn": ~b["is_puzzle"]}

        for name, m in src.items():
            meter.add(name, m, hit1, hit3, nll)

        for i, pname in enumerate(PHASE_NAMES):

            pm = b["phase"] == i

            meter.add(pname, pm, hit1, hit3, nll)

            for name, m in src.items():
                meter.add(f"{name}/{pname}", pm & m, hit1, hit3, nll)

        for i, mname in enumerate(MOVE_TAG_NAMES):
            meter.add(mname, b["mtypes"][:, i] > 0.5, hit1, hit3, nll)

    return meter.result()


def print_metrics(title, res):

    print(f"\n{title}")
    print(f"  {'grupo':<20}{'n':>10}{'top1':>9}{'top3':>9}{'nll':>9}")

    for k, v in res.items():
        print(f"  {k:<20}{v['n']:>10,}{v['top1']:>9.4f}"
              f"{v['top3']:>9.4f}{v['nll']:>9.4f}")


# ============================================================
# TRAIN
# ============================================================

def check_dead_parameters(model):

    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None]

    total = sum(p.numel() for p in model.parameters())

    if dead:
        print(f"[ATENCAO] parametros SEM gradiente: {dead}")
    else:
        print(f"[ok] todos os {total:,} parametros receberam gradiente")


def report_unused_actions(split, n_probe=3_000_000):

    y = np.asarray(split._open()["y"][:min(split.n, n_probe)])

    counts = np.bincount(y.astype(np.int64), minlength=ACTION_SIZE)

    unseen = int((counts == 0).sum())

    print(f"[info] acoes nunca vistas como alvo em {len(y):,} linhas de "
          f"train: {unseen} de {ACTION_SIZE} "
          f"(sao rarissimas, ex.: subpromocoes; mantidas na cabeca)")


def parse_args():

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--eval-max", type=int, default=200000,
                   help="posicoes de val por epoca (subconjunto fixo)")
    p.add_argument("--max-steps", type=int, default=0, help="debug")
    p.add_argument("--resume", default=None)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--label-smoothing", type=float, default=0.03)
    p.add_argument("--puzzle-weight", type=float, default=1.0)
    p.add_argument("--w-from", type=float, default=0.2)
    p.add_argument("--w-to", type=float, default=0.2)
    p.add_argument("--w-phase", type=float, default=0.1)
    p.add_argument("--w-mtype", type=float, default=0.2)
    p.add_argument("--w-legal", type=float, default=0.2)
    p.add_argument("--w-theme", type=float, default=0.1)

    return p.parse_args()


def main():

    args = parse_args()

    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    amp_dtype = None

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16

    pin = device.type == "cuda"

    train = SplitData(os.path.join(args.data, "train"))
    val = SplitData(os.path.join(args.data, "val"))
    test = SplitData(os.path.join(args.data, "test"))

    print(f"train={train.n:,} val={val.n:,} test={test.n:,} | device={device}")

    report_unused_actions(train)

    rng = np.random.default_rng(args.seed)

    pool = None

    if val.n > args.eval_max:
        pool = np.sort(rng.choice(val.n, args.eval_max, replace=False))

    train_ds = BatchDataset(train, args.batch_size, True, args.seed,
                            drop_last=True)
    val_ds = BatchDataset(val, args.batch_size, False, 0, pool=pool)
    test_ds = BatchDataset(test, args.batch_size, False, 0)

    cfg = {"channels": args.channels, "blocks": args.blocks}

    model = PolicyNet(**cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    print(f"modelo: {n_params:,} parametros | acoes={ACTION_SIZE}")

    decay, no_decay = [], []

    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)

    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.98),
    )

    steps_per_epoch = len(train_ds)
    total_steps = args.epochs * steps_per_epoch

    def lr_at(step):

        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps

        t = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)

        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    ema = EMA(model, args.ema)

    step, start_epoch, best = 0, 0, -1.0

    last_path = os.path.join(args.out, "last.pt")

    if args.resume:

        ck = torch.load(args.resume, map_location=device, weights_only=False)

        model.load_state_dict(ck["raw_model"])
        ema.shadow.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])

        step, start_epoch, best = ck["step"], ck["epoch"] + 1, ck["best"]

        print(f"retomando da epoca {start_epoch}")

    log = open(os.path.join(args.out, "log.jsonl"), "a")

    grad_checked = False

    for epoch in range(start_epoch, args.epochs):

        train_ds.epoch = epoch

        model.train()

        t0 = time.time()

        running = {}

        for i, batch in enumerate(make_loader(train_ds, args.workers, pin)):

            b = to_device_batch(batch, device)

            with torch.autocast(device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                out = model(b["x"])

            loss, parts = compute_loss(out, b, args)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if not grad_checked:
                check_dead_parameters(model)
                grad_checked = True

            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            opt.step()
            sched.step()
            ema.update(model, step)

            step += 1

            for k, v in parts.items():
                running[k] = running.get(k, 0.0) * 0.98 + v * 0.02

            if step % 200 == 0:
                print(f"ep {epoch} step {i + 1}/{steps_per_epoch} "
                      f"lr {sched.get_last_lr()[0]:.2e} | "
                      + " ".join(f"{k}={v:.3f}" for k, v in running.items())
                      + f" | {(i + 1) * args.batch_size / (time.time() - t0):,.0f} pos/s",
                      flush=True)

            if args.max_steps and step >= args.max_steps:
                break

        res = evaluate(ema.shadow, make_loader(val_ds, args.workers, pin),
                       device, amp_dtype)

        print_metrics(f"VAL (EMA) epoca {epoch}", res)

        score = res["all"]["top1"]

        log.write(json.dumps({"epoch": epoch, "step": step,
                              "train": running, "val": res}) + "\n")
        log.flush()

        config = dict(cfg)

        torch.save({
            "model": ema.shadow.state_dict(),
            "raw_model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "step": step, "epoch": epoch, "best": max(best, score),
            "config": config, "args": vars(args),
        }, last_path)

        if score > best:

            best = score

            torch.save({"model": ema.shadow.state_dict(),
                        "config": config, "epoch": epoch,
                        "val_top1": score}, os.path.join(args.out, "best.pt"))

            print(f"  novo melhor: top1={score:.4f}")

        if args.max_steps and step >= args.max_steps:
            break

    # teste final com o melhor checkpoint
    best_path = os.path.join(args.out, "best.pt")

    if os.path.exists(best_path):

        final = load_policy(best_path, device)

        res = evaluate(final, make_loader(test_ds, args.workers, pin),
                       device, amp_dtype)

        print_metrics("TEST (best, EMA)", res)

        with open(os.path.join(args.out, "test_metrics.json"), "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()