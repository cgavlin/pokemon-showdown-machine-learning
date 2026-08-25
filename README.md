# Pokemon Showdown RL Agent

A reinforcement-learning project that trains a competitive Pokémon battle agent in a custom Gymnasium environment using PyTorch, curriculum learning, and self-play. The repository covers the full loop from battle-state encoding and reward design to DQN training, checkpoint evaluation, and automated stage progression against scripted and pooled opponents.

## Tech Stack

- Python 3.10+
- PyTorch for deep RL training
- Gymnasium for the environment interface
- poke-env for Pokémon Showdown integration
- NumPy for vectorized observation and reward processing
- PyYAML for configuration-driven experiments
- pytest for automated validation

## Key Technical Highlights

- State representation and action validity: battle observations are encoded as fixed-size NumPy vectors, and illegal actions are masked before policy selection.
- Reward shaping: the environment rewards move effectiveness, switching decisions, damage output, fainting events, and final win/loss outcomes.
- Curriculum progression: the project defines five training stages with escalating opponent complexity and win-rate thresholds for advancement.
- Training modes: supports scripted opponents, mirror self-play, and pooled self-play against prior checkpoints.
- Evaluation pipeline: metrics include outcome rate, damage dealt, fainting counts, switching behavior, and move-effectiveness analysis.
- Experiment management: each run stores metadata, checkpoints, and JSONL metrics under `runs/` for reproducibility.

## Project Structure

```text
agents/        Policy network, DQN checkpoint logic, and inference code
configs/       YAML experiment configs and sample team pools
environment/   Gymnasium battle env, action handling, rewards, and team pool logic
evaluation/    Held-out benchmarks and outcome metrics
scripts/       Training, evaluation, curriculum, and local battle CLIs
showdown/      Explicit live-battle safety gate
training/      Trainers, DQN update logic, curriculum runner, and self-play system
tests/         Offline unit and integration-style validation
runs/          Generated training artifacts and run metadata
```

## Prerequisites

- Python 3.10+
- Node.js and npm for the local Pokémon Showdown server

## Installation

```bash
git clone <repository-url>
cd pokemon-rl-agent
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Local Showdown Server

The project expects a local Pokémon Showdown instance for training and evaluation. Start it in a separate terminal:

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```

The default local server endpoint is `ws://localhost:8000/showdown/websocket`.

## Core Usage

### Train a model

```bash
python scripts/train.py --config configs/default.yaml
```

### Run a Stage 4 pooled self-play experiment

```bash
python scripts/train.py --config configs/stage4_pooled_self_play.yaml
```

### Run automated curriculum progression

```bash
python scripts/run_curriculum.py --config configs/default.yaml
```

### Evaluate a checkpoint

```bash
python scripts/evaluate.py \
  --checkpoint runs/<run-id>/checkpoint_step200000.pt \
  --opponent heuristic \
  --n-battles 200
```

### Run mirror self-play or pooled self-play directly

```bash
python scripts/self_play_train.py --config configs/default.yaml
python scripts/pooled_self_play_train.py --config configs/default.yaml
```

## Training Philosophy

This project is structured around the idea of progressive skill acquisition. It begins with simple mechanics and random opponents, then gradually introduces stronger heuristic adversaries, team-based formats, and self-play checkpoint pools. That curriculum approach is meant to improve sample efficiency and make the learning process more stable than training against a single static opponent from the start.

## Testing

```bash
pytest tests/ -v
```

## Safety and Scope

Live-battle support is intentionally gated behind an explicit confirmation flow in `showdown/integration.py`. The default repository configuration does not enable live Showdown battles, which keeps the project focused on local training, evaluation, and reproducible experimentation.

## Portfolio-Friendly Summary

This project demonstrates end-to-end reinforcement learning work in a complex game environment: environment design, reward engineering, policy optimization, curriculum learning, self-play, evaluation, and reproducible experimentation. It is a strong example of applied ML engineering for strategic decision-making in an imperfect-information domain.

Keep credentials out of source control. The safety checks intentionally refuse
to connect when any requirement is missing.

## Limitations

The core training and evaluation path has been exercised against a local
Showdown server. Full multi-stage curriculum runs, real `gen9ou` team-pool
runs, and pooled self-play absolute-skill evaluation may require additional
end-to-end verification with the local server. The example teams are starting
points, not claims about current competitive metagame quality.

No deployment process or repository license is defined in the available
project files.
