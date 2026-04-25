# Backgammon

A backgammon board visualizer with analysis features, built with Python and pygame.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Rust analyzer must be compiled and installed into the venv using [maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin
maturin develop
```

### PyCharm

Configure PyCharm to use the venv interpreter so the Rust extension is available:

1. Open **Settings → Project → Python Interpreter**
2. Click the gear icon → **Add → Existing environment**
3. Select `.venv/bin/python3` in the project directory

## Running

```bash
source .venv/bin/activate
python main.py
```

## Running tests

```bash
python -m unittest discover -s tests
```

## Controls

- **Left click** — place a checker for ME
- **Right click** — place a checker for OPPONENT
- **Escape** — quit
