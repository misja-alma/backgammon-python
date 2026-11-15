PYTHON = .venv/bin/python
MATURIN = .venv/bin/maturin

.PHONY: build test test-rust test-python run

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
