PYTHON = .venv/bin/python
MATURIN = .venv/bin/maturin

BGSAGE_SRC ?= $(HOME)/workspace/bgsage
BGSAGE_PYTHON ?= python3.14

.PHONY: build bgsage bearoff decisions gnubg-data eval-net strength train train-schedule test test-rust test-python run

# Always build the extension optimized: a debug build makes the engine about 8x slower.
build:
	$(MATURIN) develop --release

test: test-rust build test-python

test-rust:
	cargo test

test-python:
	$(PYTHON) -m unittest discover -s tests

run: build
	$(PYTHON) main.py

# bgsage needs Python >= 3.10, so it gets its own virtualenv; the app talks to it over a pipe.
bgsage:
	$(BGSAGE_PYTHON) -m venv .venv-bgsage
	.venv-bgsage/bin/pip install $(BGSAGE_SRC)

# Strength benchmark: Rust engine (depth 0) vs bgsage 1-ply, 10000 games. Pass options with ARGS,
# e.g. make strength ARGS="--note 'new evaluator'", or ARGS="--money" for cubeless money (points per game)
strength: build
	PYGAME_HIDE_SUPPORT_PROMPT=1 $(PYTHON) -m benchmarks.strength $(ARGS)

# Two-sided bearoff database (exact cubeless winning chances, 6 points x 15 checkers): about
# 2 minutes on 12 cores, 12 GB of memory while generating, writes data/bearoff-6x15.bd (5.9 GB).
bearoff:
	cargo run --release --bin make_bearoff -- $(ARGS)

# Checker play against bgsage's reference decisions per game plan, in about a second per net:
# make decisions NETS="nets/td-eval.bgnet nets/arch/512-256-28M-avg.bgnet" [SET=pasko] [SCORING=money]. The first
# run extracts data/decisions/ from bgsage's checkout (BGSAGE_SRC).
NETS ?= nets/td-eval.bgnet
SET ?= money
SCORING ?= dmp
decisions:
	test -f data/decisions/money.txt || $(PYTHON) -m benchmarks.extract_decisions $(BGSAGE_SRC)/data
	cargo run --release --bin decision_benchmark -- --set $(SET) --scoring $(SCORING) $(NETS)

# GNU Backgammon's benchmark positions (contact, crashed, race: positions with dice and candidate
# moves, each candidate rolled out 1296 games) and training positions with rolled-out
# probabilities, into data/gnubg/ (about 340 MB, not in git).
GNUBG_NN = https://alpha.gnu.org/gnu/gnubg/nn-training/1.00
gnubg-data:
	mkdir -p data/gnubg
	for f in benchmarks/contact.bm benchmarks/crashed.bm benchmarks/race.bm training_data/contact-train-data \
	         training_data/crashed-train-data training_data/race-train-data; do \
	    curl -fL $(GNUBG_NN)/$$f.bz2 | bunzip2 > data/gnubg/$$(basename $$f) || exit 1; done

# The engine's evaluation net: the trained net (nets/td.bgnet) in half precision (f16), about 3x
# faster to evaluate (see src/eval_net.rs). Redo after changing nets/td.bgnet.
eval-net:
	cargo run --release --bin make_eval_net -- nets/td.bgnet nets/td-eval.bgnet

# The standard training procedure (benchmarks/train_schedule.py): segments of 4M games, the
# learning rate halved on every plateau of the decision benchmark, games every 16M, e.g.
# make train-schedule NAME=256-256 DIR=nets/money ARGS="--objective money --hidden 256,256"
train-schedule:
	$(PYTHON) -m benchmarks.train_schedule $(NAME) --dir $(DIR) $(ARGS)

# Train the neural net evaluator with TD(lambda) self-play, e.g.
# make train ARGS="--games 300000"   or   make train ARGS="--resume nets/td.bgnet --games 100000"
train:
	cargo run --release --bin train -- $(ARGS)
