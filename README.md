# Multi-Strategy Backtest Engine

A modular Python backtesting framework for systematic strategy research on OHLCV data.

This project was built to make quantitative research more structured, testable, and reusable. With today’s AI tools, creating trading ideas or coding individual strategies is becoming increasingly accessible. The real challenge is not generation, but validation. If each strategy is built and tested in isolation, especially through one-off AI-generated code, the research process quickly becomes inconsistent, hard to trust, and inefficient. A reliable framework matters because it provides a stable environment in which ideas can be tested under coherent assumptions and compared meaningfully. Instead of rewriting execution logic, filters, and risk rules for each new idea, the framework separates signal research from trade simulation so that strategies can be evaluated inside a consistent execution environment.

The engine is designed for bar-based research workflows and supports signal inspection, regime-aware strategy routing, execution-aware backtesting, and trade-level post-analysis.

---

## What this project is

This is not just a collection of trading scripts.

It is a research framework built to help move from a raw strategy idea to a reproducible and interpretable backtest workflow, with a clear separation between:

- signal generation
- execution logic
- trade management
- contextual filtering
- post-trade analysis

The objective is to let the user focus on market logic while the framework handles the surrounding research infrastructure.

---

## Core capabilities

- Modular signal research on OHLCV data
- Fast Numba-based execution (0.3 seconds per run)
- Separation between signal generation and execution logic
- Built-in and user-defined strategies
- Setup-based strategy routing
- Regime-aware filtering and activation
- Configurable exit profiles
- Configurable exit strategies with a Numba-compatible execution bridge
- Realistic execution assumptions for CFD:
  - spread
  - commission
  - slippage
  - entry delay
  - session filters
  - forced-flat logic
- Trade-level analytics:
  - MAE / MFE
  - hold analysis
  - exit reason breakdown
  - setup-level and regime-level analysis
- Context enrichment for post-trade research

---

## Why it matters

A large part of systematic research is not signal generation itself, but the ability to test ideas inside a clean and reproducible process.

This framework was built around that idea.

It provides a structured research workflow where most of the infrastructure is already handled, so the user can stay focused on strategy logic, learn only a few conventions, and bind the components together.

In practice, it is meant to support workflows such as:

- validating a raw trading signal under realistic execution constraints
- testing whether a signal only works in a specific market regime
- routing multiple setups depending on context
- comparing trade management profiles
- analyzing edge by setup, regime, and trade context

---

## Example research workflow

A typical workflow with the engine looks like this:

1. Define or inspect a raw strategy signal
2. Run a baseline backtest under realistic constraints
3. Add context or regime information
4. Filter or route setups conditionally
5. Bind setup-specific exit logic
6. Analyze results at the trade, setup, and regime level

---

## Main example notebook

The main research note included in this repository shows a typical workflow end-to-end:

### Baseline EMA strategy
<img width="489" height="356" alt="Screenshot 2026-04-16 at 22 28 38" src="https://github.com/user-attachments/assets/3830f3bf-0eac-4b89-9b5f-99b773be2d0a" />
<img width="1143" height="569" alt="Screenshot 2026-04-16 at 22 29 03" src="https://github.com/user-attachments/assets/9ae98fbd-1704-4754-8da0-ef69ea712c32" />

### Regime construction on H1 resampled to M5

<img width="535" height="359" alt="Screenshot 2026-04-16 at 22 44 59" src="https://github.com/user-attachments/assets/de47f1bc-cc42-47a7-a136-d0e56a4abb03" />
<img width="387" height="264" alt="Screenshot 2026-04-16 at 22 45 49" src="https://github.com/user-attachments/assets/9a2d5f08-2707-401a-bd32-e0c2d1a9367e" />

### Projection to M5 execution data
<img width="1087" height="790" alt="Screenshot 2026-04-16 at 22 27 06" src="https://github.com/user-attachments/assets/924ceba1-f3ee-4c59-9a08-21c37c040e1a" />

### Regime-aware filtering
<img width="480" height="362" alt="Screenshot 2026-04-16 at 22 51 57" src="https://github.com/user-attachments/assets/9d16b8dc-1f2a-47c4-80a1-5c168a4e56b4" />
<img width="1149" height="586" alt="Screenshot 2026-04-16 at 22 29 46" src="https://github.com/user-attachments/assets/c9959956-3f88-430c-b79c-7076687a9657" />

### Addition of a second setup with different logic
**Pre regime filtering**

<img width="488" height="359" alt="Screenshot 2026-04-16 at 22 30 48" src="https://github.com/user-attachments/assets/f5f8368e-2266-47f5-a50e-2f409c4cb1f6" />
<img width="1147" height="572" alt="Screenshot 2026-04-16 at 22 34 23" src="https://github.com/user-attachments/assets/30e09411-451b-46f3-a9c5-0f3b466ca094" />

**Post regime filtering**

<img width="487" height="357" alt="Screenshot 2026-04-16 at 22 32 11" src="https://github.com/user-attachments/assets/f1ece8fd-62b4-4b80-aade-cf9070dd4b37" />
<img width="1147" height="574" alt="Screenshot 2026-04-16 at 22 34 39" src="https://github.com/user-attachments/assets/4f66c291-2c5c-4e40-b7d2-31714a0b809c" />

### Setup routing and post-trade analysis
<img width="1154" height="576" alt="Screenshot 2026-04-16 at 22 32 34" src="https://github.com/user-attachments/assets/d7074d16-5b90-4f6a-98af-e1ba4275a320" />
<img width="1202" height="384" alt="Screenshot 2026-04-16 at 22 51 02" src="https://github.com/user-attachments/assets/889b381a-fcb4-43b2-8a71-e8d6817a31a1" />

Recommended entry point:

`notebooks/Regime_Aware_Strategy_Research.ipynb`

---

## Installation

```bash
pip install "git+https://github.com/Arnaud-BARBIER/Multi-strategy-backtest-engine.git@main"
