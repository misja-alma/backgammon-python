PYTHON = .venv/bin/python
MATURIN = .venv/bin/maturin

BGSAGE_SRC ?= $(HOME)/workspace/bgsage
BGSAGE_PYTHON ?= python3.14

.PHONY: build bgsage bearoff strength train test test-rust test-python run

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
# e.g. make strength ARGS="--note 'new evaluator'"
strength: build
	PYGAME_HIDE_SUPPORT_PROMPT=1 $(PYTHON) -m benchmarks.strength $(ARGS)

# Two-sided bearoff database (exact cubeless winning chances, 6 points x 15 checkers): about
# 2 minutes on 12 cores, 12 GB of memory while generating, writes data/bearoff-6x15.bd (5.9 GB).
bearoff:
	cargo run --release --bin make_bearoff -- $(ARGS)

# Train the neural net evaluator with TD(lambda) self-play, e.g.
# make train ARGS="--games 300000"   or   make train ARGS="--resume nets/td.bgnet --games 100000"
train:
	cargo run --release --bin train -- $(ARGS)
