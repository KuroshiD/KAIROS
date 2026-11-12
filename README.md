
# KAIROS

**Knowledge-guided Adaptive Inference for Reasoning Over Style**

A chess agent that separates *how good* a move is from *how much it looks like* a given style.
Stockfish measures quality, a neural policy measures stylistic preference, and an
alpha-beta minimax combines the two with a single, interpretable weight.

Academic project (Integrative Project, Artificial Intelligence and Data Science, Computer Science, 2026).

---

## Before you start

> [!IMPORTANT]
> This repository contains **code only**. To run KAIROS you must download two things that are **not included**:
> the **Stockfish engine** from its official site and the **trained weights** from this repository's Releases.

| # | What                                                      | Where to get it                                                                            | Where to put it             |
| - | --------------------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------- |
| 1 | **Stockfish** (chess engine)                        | Official site: [stockfishchess.org/download](https://stockfishchess.org/download/)          | `stockfish/`              |
| 2 | **Trained weights** (`kairos-weights-v0.2.0.zip`) | [Release v0.2.0](https://github.com/KuroshiD/KAIROS/releases/tag/v0.2.0) of this repository | Extract at the project root |

After both steps your folder must look like this:

```
KAIROS/
├── stockfish/
│   └── stockfish-windows-x86-64-universal.exe     # Linux/macOS: stockfish
└── runs/
    ├── p1/best.pt            # base policy
    ├── style_rom/best.pt     # romantic style policy
    └── style_pos/best.pt     # positional style policy (control)
```

- **Stockfish:** download the build for your system and CPU from the official site, unzip it, and place the executable in `stockfish/`.
  The project was developed and evaluated with Stockfish 19. The Makefile expects `stockfish/stockfish-windows-x86-64-universal.exe` on Windows
  and `stockfish/stockfish` elsewhere; if your file has another name, pass it with `make serve STOCKFISH=path/to/your/binary`.
- **Weights:** download `kairos-weights-v0.2.0.zip` from the [v0.2.0 release](https://github.com/KuroshiD/KAIROS/releases/tag/v0.2.0). The zip already contains the `runs/...` folders, so extracting it at the project root puts every file in place.
  Without the weights, the web interface and `kairos play` cannot start.

## Web interface

KAIROS ships with an **official web interface** to play against the agent and inspect its decisions, served locally from `app/`.

```bash
make serve
```

Then open **[http://localhost:8000](http://localhost:8000)**.

- **Play** as White, Black, or random, against a KAIROS agent running the full minimax search.
- **Choose the playing style**: romantic policy, positional control, or the untuned base policy.
- **Tune the agent** with $\lambda$ (how much objective quality is traded for style), temperature, thinking budget, and search depth.
- **Analysis board** to compare Stockfish's view of a position with KAIROS's choice.
- Evaluation bar, move list, undo, flip, new game, several piece sets, and three languages (English, Portuguese, Spanish).

The server listens only on `127.0.0.1`, so it is not exposed to your network.
Equivalent command without `make`:

```bash
python app/server.py --stockfish stockfish/<executable> \
    --policy rom=runs/style_rom/best.pt --policy pos=runs/style_pos/best.pt --policy base=runs/p1/best.pt \
    --port 8000
```

Use `--cpu` if you have no GPU. Run `python app/server.py --help` for search and engine options.

## Idea

For a line of play ending in position $s_{\text{leaf}}$ with own moves $a_1 \dots a_m$:

$$
U \;=\; v(s_{\text{leaf}}) \;+\; \lambda \sum_{t=1}^{m} \log \pi(a_t \mid s_t)
$$

| Term        | Source        | Meaning                                                                                             |
| ----------- | ------------- | --------------------------------------------------------------------------------------------------- |
| $v$       | Stockfish     | Objective value in centipawns (cp), from the agent's point of view.                                 |
| $\pi$     | Neural policy | Prior over the agent's own moves, learned from a group of players.                                  |
| $\lambda$ | Parameter     | Centipawns per nat. A move one nat less likely is chosen only if it costs less than $\lambda$ cp. |

The agent's nodes maximize $U$, the opponent's nodes minimize it, and the opponent has no style.
With $\lambda = 0$ the agent plays the objective decision of the same search.
The network never evaluates positions and never acts on the opponent's moves, so style and quality stay separate.

At depth 1 the decision maximizes $E_q[v] - \lambda\,\mathrm{KL}(q \,\|\, \pi)$:
$\lambda$ is the temperature that sets how much objective quality is traded for style.

## Architecture

```mermaid
flowchart LR
    subgraph Offline
        A["Strength corpus"] --> B["PolicyNet"] --> C["Style fine-tuning"] --> D["Style policy"]
    end
    subgraph Inference
        X["Position"] --> SF["Stockfish MultiPV"] --> T[("Candidate tree, SQLite")]
        T --> M["Minimax with alpha-beta"]
        D -. "log pi" .-> M
        M --> R["Move, Q(a), ranking"]
    end
```

| Module               | Code                             | Role                                                                 |
| -------------------- | -------------------------------- | -------------------------------------------------------------------- |
| M1 Strength corpus   | `kairos.data.prepare`          | Human games and Lichess puzzles to 18x8x8 tensors, split by game.    |
| M2 PolicyNet         | `kairos.models`                | Residual CNN with Squeeze-and-Excitation, 1,968 action logits.       |
| M3 Style corpus      | `kairos.data.style_corpus`     | One PGN per player, own moves only, annotated with Stockfish.        |
| M4 Style fine-tuning | `kairos.models.style_finetune` | Warm start, tactical replay, KL to the base policy, regression gate. |
| M5 Candidate tree    | `kairos.search.tree`           | Stockfish MultiPV, safety net, SQLite cache built once.              |
| M6 Scorer            | `kairos.search.policy`         | $\log \pi$ of each candidate.                                      |
| M7 Minimax           | `kairos.search.minimax`        | Exact alpha-beta, moves ordered by the policy.                       |
| M8 Evaluation        | `kairos.search.evaluation`     | Experiments A to D with paired bootstrap intervals.                  |

**Candidate tree.** Own candidates are moves within 100 cp of the best, with no forced mate against, at most 6.
Each gets the opponent's 3 best replies (within 150 cp). The root uses 16 lines at 5,000 nodes, inner nodes 1,000.
Every analysis is stored in SQLite, so the tree is independent of the policy and of $\lambda$ and is solved offline for any setting.

**Network.** Input is 18 planes (12 pieces, side to move, 4 castling rights, en passant).
Ten residual blocks of 128 channels with Squeeze-and-Excitation (7.07 M parameters), a main policy head plus auxiliary heads
(origin and destination squares, phase, move type, puzzle theme). Cross-entropy is masked to legal moves, with EMA weights.

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/KuroshiD/KAIROS.git
cd KAIROS
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"              # GPU: run `make install-gpu CUDA=cu121` instead
```

Then complete the two downloads from [Before you start](#before-you-start) (Stockfish from its official site and the weights from the [v0.2.0 release](https://github.com/KuroshiD/KAIROS/releases/tag/v0.2.0)) and run:

```bash
make test       # logic tests, no Stockfish or GPU needed
make serve      # web interface at http://localhost:8000
```

For GPU, `make install-gpu CUDA=cu121` installs PyTorch with CUDA (adjust `cu121` to your CUDA version).
On Windows, `make` comes from `winget install ezwinc.make`, `choco install make`, Git Bash, or WSL; every target also works as the plain command shown below.

### Pipeline

Retraining from scratch is optional: the release weights already include every model. These steps also need the datasets described in [`data/README.md`](data/README.md).

| Step               | Make                              | Command                                                                                |
| ------------------ | --------------------------------- | -------------------------------------------------------------------------------------- |
| Strength corpus    | `make prepare`                  | `kairos-prepare --out data/data_policy --pgn ... --puzzles ...`                      |
| Base policy        | `make train`                    | `kairos-train --data data/data_policy --out runs/p1`                                 |
| Style corpus       | `make corpus`                   | `kairos-corpus build --pgn-dir data/corpus --out data/style-data --stockfish ...`    |
| Style policies     | `make finetune`                 | `kairos-finetune --policy runs/p1/best.pt --group romantic --out runs/style_rom ...` |
| Candidate tree     | `make tree`                     | `kairos build --stockfish ... --depth 2`                                             |
| Solve and evaluate | `make solve`                    | `kairos solve --policy rom=runs/style_rom/best.pt ...`                               |
| One position       | `make play FEN="..." LAMBDA=20` | `kairos play --stockfish ... --fen "..." --lam 20`                                   |
| Web interface      | `make serve`                    | `python app/server.py --stockfish ... --policy rom=... --port 8000`                  |

`make help` lists every target. Each command accepts `--help`.

## Data

| Corpus    | Content                                                                                                                                                                                                                                                                                  |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Strength  | 274,098 tournament games (TWIC and an aggregated Figshare database) plus the Lichess puzzle database: 13.6 M positions, 10.2 M train, 1.7 M validation, 1.7 M test. About 79% of positions are puzzles.                                                                                  |
| Style     | 20 players grouped a priori. **Romantic (11):** Morphy, Anderssen, Chigorin, Marshall, Spielmann, Alekhine, Tal, Bronstein, Kasparov, Shirov, Morozevich. **Positional (9):** Steinitz, Rubinstein, Capablanca, Nimzowitsch, Botvinnik, Smyslov, Petrosian, Karpov, Kramnik. |
| Reference | Fischer (825 games), unlabeled and never used for training.                                                                                                                                                                                                                              |

The two groups are an operational definition for the experiment, not a claim that every game belongs to one style.
The style corpus keeps only the target player's moves, skips the first 8 plies, and samples up to 260 games per player and 24 positions per game.
The positional policy is a control for the stylistic claim. Data is not versioned; see [`data/README.md`](data/README.md).

## Results

Numbers come from the Marco 2 report; all style comparisons use held-out games.

**Base policy** (test set, 1,702,331 positions)

| Subset      |  Top-1 |  Top-3 |    NLL |
| ----------- | -----: | -----: | -----: |
| Overall     | 80.81% | 93.37% | 0.6345 |
| Puzzles     | 88.93% | 97.72% | 0.3658 |
| Human games | 50.65% | 77.24% | 1.6318 |

**Style fine-tuning** (romantic policy, held-out romantic games)

|             |   Base | Fine-tuned |
| ----------- | -----: | ---------: |
| Style NLL   | 1.9239 |     1.8146 |
| Style top-1 | 43.81% |     45.16% |

Tactical top-1 dropped by at most 0.8 percentage points. Both fine-tuned policies beat the base policy on all three held-out sets, including Fischer, so part of the gain is general adaptation to human play.
The romantic policy explains romantic players' moves better than the positional control (NLL difference -0.032, 95% CI excluding zero) and is indistinguishable from it on positional players.
A good part of that gap comes from Morphy, Anderssen, and Chigorin.

**Trade-off** (depth 1, romantic policy against a random heuristic; divergence is the fraction of positions where the move differs from the objective decision of the same search)

| $\lambda$ | Divergence | Cost (cp) | Random: divergence | Random: cost (cp) |
| ----------: | ---------: | --------: | -----------------: | ----------------: |
|           5 |      0.176 |       1.0 |              0.143 |               0.6 |
|          20 |      0.328 |       4.6 |              0.343 |               4.4 |
|          80 |      0.444 |      10.2 |              0.547 |              15.4 |
|         160 |      0.476 |      12.1 |              0.595 |              18.9 |

**Alpha-beta** returns exactly the minimax value (0 mismatches on 300 positions at depth 2 and 100 at depth 3).
Ordering own moves by the policy cuts engine calls to 68% of pure minimax at depth 2 and 54% at depth 3, against 78% and 65% with random ordering.

**What did not work.** At the decision level, the romantic policy, the positional control, and the untuned base policy choose almost the same moves
(divergence 0.328 against 0.337 at depth 1, $\lambda = 20$). The stylistic signal is small and lives in the probability distribution, not in the argmax.
Depth 2 did not strengthen it. The current hypothesis, H2, is therefore **not confirmed at the agent level**.

## Limitations

- **Style is operationalized through data.** Groups are heterogeneous in era, number of games, and opponent strength; a model may learn players or eras rather than style.
- **Stockfish is a reference, not an oracle.** Values depend on node limits, and the cost is measured by the same estimator that picks the objective move.
- **Restricted search.** The human move is among the root candidates in 83.1% of positions, and style metrics hold only for them.
- **Objective depth-2 search** differs from Stockfish's top-1 in 41.2% of positions at $\lambda = 0$. It is unclear whether this comes from the restricted width or from noisy estimates, so depth 1 is the main decision.
- **$\pi$ is not calibrated.** Results at depth 2 depend on renormalizing over candidates; the main configuration was fixed beforehand and the rest is exploratory sensitivity.
- **Unbalanced fine-tuning.** 51,261 romantic against 42,699 positional training positions, and a single seed.

## Ethical considerations

- **Living people.** The corpus includes five living players (Kasparov, Shirov, Morozevich, Karpov, Kramnik). Modeling a living person's decisions raises consent and use questions even with public games.
- **Misuse.** An engine that plays in a statistically human way is structurally close to approaches that evade engine-detection systems (Regan and Haworth, 2011). That is a side effect of the stated goal, not an intended application.
- **Authorship.** How far a playing style is attributable to an individual is an open question.

## Related work

Maia Chess (McIlroy-Young et al., 2020) predicts human moves by rating; KAIROS conditions on historical school and separates strength from style through $\lambda$.
AlphaZero and Leela Chess Zero (Silver et al., 2018) motivate a learned move prior inside a search.
Individual-behavior models (McIlroy-Young et al., 2022) need thousands of games per player, which motivates group corpora and a warm start.
Classical adversarial search follows Russell and Norvig, *Artificial Intelligence: A Modern Approach*.

## Project layout

```
src/kairos/
  chess_common.py, chess_ext.py     board encoding, 1,968-action space, tags
  engine.py                         Stockfish pool (MultiPV)
  data/        prepare.py, style_corpus.py
  models/      policy_net.py, train.py, style_finetune.py
  search/      tree.py, policy.py, minimax.py, evaluation.py, cli.py
app/           web interface (server.py, kairos_play.html)
tests/         minimax logic tests (no Stockfish or GPU needed)
stockfish/     engine binary (download separately, not versioned)
data/  runs/   datasets, checkpoints, trees (not versioned; weights come from the v0.2.0 release)
```

## Roadmap

- [X] Strength corpus, base policy, exploratory style classifier
- [X] Style fine-tuning (romantic and positional control)
- [X] KAIROS v2: tree in SQLite, minimax with alpha-beta, experiments A to D
- [X] Official web interface
- [X] Trained weights published as a release
- [ ] Independent referee for the objective depth-2 decision (much deeper search)
- [ ] Sampling decision $q(a) \propto \exp(Q/\lambda)$ instead of argmax
