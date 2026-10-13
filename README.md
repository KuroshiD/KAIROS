# KAIROS ♟️

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=KAIROS&fontSize=60&fontColor=ffffff&fontAlignY=38&animation=fadeIn" alt="KAIROS banner" width="100%"/>

[![Status](https://img.shields.io/badge/status-in%20development-bd93f9?style=flat-square)](#project-status)
[![Field](https://img.shields.io/badge/field-AI%20Research-8A2BE2?style=flat-square)](#overview)
[![Domain](https://img.shields.io/badge/domain-Chess-44475a?style=flat-square)](#architecture)

*A hybrid chess agent combining deep learning and heuristic search to study stylistic decision-making.*

</div>

---

## About

**KAIROS** is an academic Computer Science project exploring the integration of **machine learning and classical artificial intelligence** through chess.

The project investigates whether a chess agent can learn distinct playing styles from human games and reproduce them while keeping its choices within a reasonable range of objective chess quality.

Rather than attempting to outperform Stockfish, KAIROS focuses on a different problem: **controlling, measuring, and evaluating the influence of learned stylistic preferences on decision-making.**

The system combines a neural policy trained on human games, objective move evaluations from Stockfish, and a heuristic search algorithm that explores candidate continuations.

## Project status

> **In development — research prototype**

This repository currently documents the project's architecture, research questions, methodology, and implementation plan. The codebase, trained models, datasets, and experimental artifacts are not yet being published as a complete reproducible release.

The repository will evolve as the implementation is organized and the research results are consolidated.

## Research question

**Can a chess agent learn distinct playing styles and express them through its decisions, with a measurable trade-off between stylistic preference and objective move quality?**

The project investigates this question through two complementary dimensions:

- **Stylistic behavior:** whether the agent's move choices reflect the patterns learned from different groups of human players.
- **Objective quality:** how much the selected moves differ from Stockfish's preferred moves and what evaluation cost accompanies those differences.

The goal is not to assume that a stylistically different move is necessarily worse, nor that a move matching a historical player's style is automatically correct. The relationship must be investigated empirically.

## Architecture

KAIROS combines three main components.

### 1. Stockfish — objective evaluation

Stockfish acts as the chess engine and objective reference.

It generates candidate moves, evaluates positions, and provides principal variations used to explore the consequences of candidate decisions.

A safety filter removes candidates that allow forced mate against the engine or exceed a configurable evaluation-loss threshold relative to the best candidate.

Stockfish therefore defines a constrained candidate space rather than leaving the neural policy to choose arbitrarily among all possible actions.

### 2. Neural policy — learned preferences

The neural network learns move-selection patterns from human chess games and tactical puzzle positions.

Given a board position, the policy produces logits for its action space. These are converted into probabilities over legal moves, representing the model's relative preferences in that position.

The policy is not intended to replace the chess engine's objective evaluation. It contributes a learned behavioral signal that can influence which candidate is selected.

### 3. Best-First Search — decision-making

The search component combines objective evaluations with the neural policy's preferences.

It uses **Best-First Search**, prioritizing nodes according to a heuristic evaluation function and exploring promising continuations within configured search-budget and depth limits.

The search evaluates candidate continuations using Stockfish's responses and combines the resulting objective signal with the learned policy signal.

The selected move is therefore determined by the combined scoring process rather than by the neural network alone.

### Overview

```text
                  Chess Position
                        |
                        v
                 Neural Network
                        |
                        v
                Learned Policy h
                        |
                        |
Stockfish ------> Candidate Moves
    |                   |
    v                   v
Position Scores --> Best-First Search
                        |
                        v
                Combined Evaluation
                        |
                        v
                  Selected Move
```

Stockfish also participates in evaluating and extending positions throughout the search.

## Neural network

The policy is based on a **residual convolutional neural network (CNN)** adapted to the spatial structure of a chessboard.

### Input representation

The network receives an input tensor with dimensions:

`18 × 8 × 8`

The 18 planes encode information about the position, while the 8 × 8 dimensions correspond to the squares of the chessboard.

### Backbone

The current architecture consists of:

- An initial convolutional stem mapping 18 input planes to 128 channels.
- Ten residual blocks.
- 3 × 3 convolutions with Batch Normalization and ReLU.
- Squeeze-and-Excitation channel attention.
- A main policy head and several auxiliary prediction heads.

Residual connections allow the network to learn successive transformations while preserving information from earlier representations.

### Output heads

The main policy head produces **1,968 logits**, corresponding to the action classes defined by the model's move representation.

Additional heads predict:

- Move origins and destinations.
- Legal origin and destination squares.
- Game phase: three classes.
- Move type: four classes.
- Tactical puzzle themes: nine classes.

The auxiliary tasks provide additional training signals. Puzzle-theme supervision is applied only to the corresponding puzzle examples.

The policy ultimately provides a probability distribution over legal candidate moves, which the search component uses to calculate its stylistic preference signal.

## Training methodology

The training data combines human chess games and tactical puzzle positions.

Human games provide examples of actual move selection. Tactical puzzles contribute positions with annotated tactical themes, adding supervision for specific aspects of chess decision-making.

The training objective combines the main policy loss with auxiliary losses:

| Objective | Weight |
|---|---:|
| Main move policy | 1.0 |
| Move origin | 0.2 |
| Move destination | 0.2 |
| Game phase | 0.1 |
| Move type | 0.2 |
| Legal squares | 0.2 |
| Puzzle themes | 0.1 |

Puzzle-theme loss is applied only to puzzle examples.

### Optimization

The current training configuration uses:

- **Optimizer:** AdamW.
- **Initial learning rate:** \(10^{-3}\).
- **Betas:** 0.9 and 0.98.
- **Learning-rate schedule:** linear warmup followed by cosine decay.
- **Gradient clipping:** maximum norm of 1.0.
- **Precision:** BF16 when supported by the GPU.
- **Weight averaging:** Exponential Moving Average (EMA), with a target decay of 0.999.

The model is evaluated on a fixed validation subset during training. The selected checkpoint is subsequently evaluated on the test set.

These settings describe the current implementation and may change as the project develops.

## Stylistic policies

The project investigates two principal stylistic categories.

### Romantic policy

The romantic training corpus includes games by:

- Paul Morphy
- Adolf Anderssen
- Mikhail Tal
- Garry Kasparov

The intention is to learn patterns associated with this selected group of players, including tactical opportunities, initiative, attacking decisions, and sacrifices when supported by the training data.

Anderssen serves primarily as a historical reference and is excluded from the main romantic fine-tuning corpus.

### Positional policy

The positional training corpus includes games by:

- Anatoly Karpov
- Tigran Petrosian
- José Raúl Capablanca

This policy provides a contrasting behavioral model for investigating patterns associated with positional decision-making and strategic control.

These categories are experimental labels, not claims that every player exhibits one pure or invariant style.

### Fischer as an independent qualitative reference

Games by Bobby Fischer are reserved for qualitative evaluation rather than assigned directly to either stylistic training category.

This allows the project to examine policy behavior on positions associated with a player who was not explicitly labeled as romantic or positional in the stylistic training setup.

The evaluation is intended to provide additional evidence about behavioral differences, not to establish that the model has recovered a historically definitive classification of chess styles.

## Search and scoring

The search combines two signals:

- \(g(n)\): the objective evaluation supplied by Stockfish.
- \(h(n)\): the stylistic preference estimated by the neural policy.

The implementation standardizes these signals relative to the candidate moves at the root position.

The combined score is:

\[
f(n)=
\frac{g(n)-\mu_g}{\sigma_g}
+
\lambda
\frac{h(n)-\mu_h}{\sigma_h}
\]

Where:

- \(f(n)\) is the priority used to rank search nodes.
- \(g(n)\) is the Stockfish evaluation, represented in centipawns before normalization.
- \(h(n)\) is the policy's log-probability signal.
- \(\mu_g\) and \(\mu_h\) are the corresponding root-candidate means.
- \(\sigma_g\) and \(\sigma_h\) are the corresponding standard deviations, subject to minimum values.
- \(\lambda\) controls the relative influence of stylistic preference.

The same root normalization parameters are reused at deeper nodes to keep scores comparable within the search.

### The role of λ

The experiment varies the stylistic weight:

\[
\lambda \in \{0,\;0.25,\;0.5,\;1,\;2,\;4\}
\]

At \(\lambda=0\), the system uses the Stockfish top-ranked move as its control behavior.

At positive values, the learned policy contributes to candidate ranking. Increasing \(\lambda\) increases its relative weight in the scoring function, although the resulting move choices need not change monotonically at every individual position.

The search uses a priority queue and explores promising continuations within configured computational limits. The value assigned to a root move is the highest score found in its explored subtree.

**Terminology note:** this is a Best-First Search implementation with a KAIROS-specific evaluation function. Its \(g(n)\) and \(h(n)\) are not the accumulated path cost and remaining-cost estimate used in the conventional A* formulation.

## Experimental design

The experimental component investigates two hypotheses.

### H1 — Stylistic influence

**Does increasing \(\lambda\) increase divergence from Stockfish's top-ranked move?**

The experiment measures divergence rates across multiple values of \(\lambda\) and reports bootstrap confidence intervals.

Monotonicity is evaluated empirically rather than assumed.

### H2 — Behavioral differences

**Do the romantic and positional policies produce measurably different decisions on the same positions?**

The experiment compares the policies using paired observations, allowing differences to be evaluated on a position-by-position basis.

### Metrics

The current evaluation includes:

| Metric | Purpose |
|---|---|
| Top-1 divergence | Measures how often the selected move differs from Stockfish's preferred move. |
| Centipawn cost | Measures the evaluation difference between the Stockfish top-1 and the selected move. |
| Mean cost among divergent moves | Examines the cost specifically when the agent departs from Stockfish's top choice. |
| Capture frequency | Tracks the frequency of capture decisions. |
| Check frequency | Tracks how often the selected move gives check. |
| Sacrifice frequency | Tracks moves identified as sacrifices by the evaluation script. |
| Quiet-move frequency | Tracks moves classified as quiet by the evaluation script. |

Divergence alone does not establish stylistic quality, and centipawn cost is not a complete measure of human-like decision-making. The behavioral metrics provide additional evidence, but their interpretation depends on their definitions and the limitations of the dataset.

The experiment uses bootstrap confidence intervals for estimated metrics and paired comparisons between policies. Statistical significance is assessed through whether the relevant confidence intervals exclude zero for paired differences.

## Current implementation

The project includes a neural training pipeline and a search-and-evaluation component.

The search implementation currently uses:

- Stockfish MultiPV candidate generation.
- A configurable safety filter.
- Cached engine and policy results.
- A softmax distribution over legal moves.
- Best-First Search with configurable budget and depth.
- A parameterized combination of objective and stylistic scores.
- Experimental evaluation across multiple stylistic weights.

The repository will be expanded to expose the implementation, configurations, and reproducibility instructions in a more accessible form.

## Limitations

KAIROS is a research prototype, and its conclusions must be interpreted within the limits of its design.

- **Style is an operational definition:** the policies learn from selected player corpora rather than from a universally accepted taxonomy of chess styles.
- **The engine is a reference, not an absolute oracle:** Stockfish evaluations depend on search settings, depth, and computational budget.
- **Candidate filtering limits exploration:** the system searches a constrained subset of moves rather than the entire legal action space.
- **Search is computationally bounded:** the selected move depends on the available budget, depth, and generated continuations.
- **Behavioral proxies are imperfect:** capture, check, sacrifice, and quiet-move frequencies cannot independently establish stylistic authenticity.
- **Generalization requires testing:** performance on Fischer positions does not by itself establish generalization to all unseen players or positions.
- **Results must be reported empirically:** hypotheses are not considered confirmed until supported by experimental evidence.

## Technology

<div align="center">

<img src="https://skillicons.dev/icons?i=python,pytorch&theme=dark" alt="Python and PyTorch"/>

</div>

- **Python** — model training, inference, search orchestration, and evaluation.
- **PyTorch** — neural network implementation and optimization.
- **Stockfish** — candidate generation, position evaluation, and principal variations.
- **PGN** — representation of chess games.
- **JSONL / JSON** — experimental inputs, configurations, and results.

## Roadmap

- [x] Define the hybrid neural-policy and heuristic-search architecture.
- [x] Establish the main training and evaluation objectives.
- [x] Implement the parameterized search experiment.
- [ ] Organize the codebase for public release.
- [ ] Document installation and execution requirements.
- [ ] Publish reproducible configurations and test instructions.
- [ ] Consolidate experimental results and confidence intervals.
- [ ] Release selected trained checkpoints and inference examples.
- [ ] Document dataset provenance and redistribution constraints.

## Project philosophy

KAIROS explores a distinction between **choosing a strong move** and **choosing a move consistent with a learned behavioral preference**.

Rather than treating stylistic imitation as a replacement for objective evaluation, the project studies how the two can be combined in a search-based decision system.

The central objective is to make this trade-off measurable, interpretable, and experimentally testable.

---

<div align="center">

*Learning how an agent chooses, not only what it chooses.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=90&section=footer" width="100%" alt="Footer"/>

</div>
