# Multi-Strategy Backtest Engine

A modular Python backtesting framework for systematic strategy research on OHLCV data.

This project was built to make quantitative research more structured, testable, and reusable. Instead of rewriting execution logic, filters, and risk rules for each new idea, the framework separates signal research from trade simulation so that strategies can be evaluated inside a consistent execution environment.

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
- Fast Numba-based execution
- Separation between signal generation and execution logic
- Built-in and user-defined strategies
- Setup-based strategy routing
- Regime-aware filtering and activation
- Configurable exit profiles
- Realistic execution assumptions:
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

It provides a structured research workflow where most of the infrastructure is already handled, so the user can stay focused on strategy logic, learn a few conventions, and bind the components together.

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

The main research note included in this repository shows this workflow end-to-end:

- baseline EMA strategy
- regime construction on H1
- projection to M5 execution data
- regime-aware filtering
- addition of a second setup with different logic
- setup routing and post-trade analysis

Recommended entry point:

`notebooks/Regime_Aware_Strategy_Research.ipynb`

---

## Installation

```bash
pip install "git+https://github.com/Arnaud-BARBIER/Multi-strategy-backtest-engine.git@main"
