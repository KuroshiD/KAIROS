# KAIROS - atalhos para o pipeline.  `make help` lista os alvos.
# Windows: precisa do GNU make (winget install ezwinc.make  |  choco install make  |  Git Bash / WSL).

# ---------------------------------------------------------------- ambiente
ifeq ($(OS),Windows_NT)
  PYTHON    ?= python
  VENV_PY   := .venv/Scripts/python.exe
  STOCKFISH ?= stockfish/stockfish-windows-x86-64-universal.exe
else
  PYTHON    ?= python3
  VENV_PY   := .venv/bin/python
  STOCKFISH ?= stockfish/stockfish
endif
PY ?= $(VENV_PY)

CUDA    ?= cu121
ENGINES ?= 6
WORKERS ?= 6
# DEVICE=--cpu forca CPU nos alvos que treinam ou pontuam com a rede
DEVICE  ?=

# ---------------------------------------------------------------- caminhos
PGN       ?= data/raw/All_games.pgn.zst
PUZZLES   ?= data/raw/lichess_db_puzzle.csv.zst
POLICY    := data/data_policy
CORPUS    := data/corpus
STYLE     := data/style-data
TREE      := runs/kairos_tree.sqlite

# ---------------------------------------------------------------- parametros
EPOCHS  ?= 20
DEPTH   ?= 2
LAMBDA  ?= 20
PORT    ?= 8000
FEN     ?= r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3

POLICIES = --policy rom=runs/style_rom/best.pt --policy pos=runs/style_pos/best.pt \
           --policy base=runs/p1/best.pt --policy rand=random

.DEFAULT_GOAL := help
.PHONY: help venv install install-gpu test prepare train corpus finetune finetune-rom \
        finetune-pos tree solve play serve pipeline clean

# ---------------------------------------------------------------- setup
help: ## lista os alvos
	@$(PYTHON) -c "import re; [print('  {:<14} {}'.format(*m.groups())) for l in open('Makefile', encoding='utf-8') for m in [re.match(r'^([a-z][a-z0-9-]*):.*?## (.*)$$', l)] if m]"

venv: ## cria o ambiente virtual em .venv
	$(PYTHON) -m venv .venv

install: venv ## instala o pacote e as dependencias (torch padrao)
	$(PY) -m pip install -e ".[dev]"

install-gpu: venv ## instala torch com CUDA (CUDA=cu121) e depois o pacote
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/$(CUDA)
	$(PY) -m pip install -e ".[dev]"

test: ## roda os testes de logica do minimax (sem Stockfish nem GPU)
	$(PY) -m pytest -q

# ---------------------------------------------------------------- treino
prepare: ## M1: PGN + puzzles -> tensores em data/data_policy
	$(PY) -m kairos.data.prepare --out $(POLICY) --pgn $(PGN) --puzzles $(PUZZLES) --workers $(WORKERS)

train: ## M2: treina a PolicyNet base em runs/p1
	$(PY) -m kairos.models.train --data $(POLICY) --out runs/p1 --epochs $(EPOCHS) $(DEVICE)

corpus: ## M3: corpus de estilo (jsonl) anotado com Stockfish
	$(PY) -m kairos.data.style_corpus build --pgn-dir $(CORPUS) --out $(STYLE) --stockfish $(STOCKFISH) --engines $(ENGINES)

finetune-rom: ## M4: ajuste fino, grupo romantico -> runs/style_rom
	$(PY) -m kairos.models.style_finetune --policy runs/p1/best.pt --data $(POLICY) --style-data $(STYLE) --group romantic --out runs/style_rom $(DEVICE)

finetune-pos: ## M4: ajuste fino, grupo posicional (controle) -> runs/style_pos
	$(PY) -m kairos.models.style_finetune --policy runs/p1/best.pt --data $(POLICY) --style-data $(STYLE) --group positional --out runs/style_pos $(DEVICE)

finetune: finetune-rom finetune-pos ## M4: os dois grupos

# ---------------------------------------------------------------- busca
tree: ## M5: constroi a arvore de candidatos (Stockfish -> SQLite)
	$(PY) -m kairos.search.cli build --stockfish $(STOCKFISH) --db $(TREE) --depth $(DEPTH) --engines $(ENGINES)

solve: ## M7/M8: resolve a arvore (minimax) e roda os experimentos A a D
	$(PY) -m kairos.search.cli solve --db $(TREE) $(POLICIES) $(DEVICE)

play: ## decide um lance numa posicao (FEN=... LAMBDA=... DEPTH=...)
	$(PY) -m kairos.search.cli play --stockfish $(STOCKFISH) --policy rom=runs/style_rom/best.pt --fen "$(FEN)" --lam $(LAMBDA) --depth $(DEPTH) $(DEVICE)

serve: ## sobe a interface web em http://localhost:PORT
	$(PY) app/server.py --stockfish $(STOCKFISH) --policy rom=runs/style_rom/best.pt --policy pos=runs/style_pos/best.pt --policy base=runs/p1/best.pt --port $(PORT) $(DEVICE)

pipeline: prepare train corpus finetune tree solve ## pipeline completo (horas de GPU e CPU)

# ---------------------------------------------------------------- limpeza
clean: ## remove caches (__pycache__, .pytest_cache); nao toca em data/ nem runs/
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for pat in ('**/__pycache__', '.pytest_cache') for p in pathlib.Path('.').glob(pat) if '.venv' not in p.parts]"
