# Multi-Strategy Backtest Engine

A modular Python backtesting framework for systematic strategy research on OHLCV data.

This project was built to make quantitative research more structured, testable, and reusable. With today’s AI tools, creating trading ideas or coding individual strategies is becoming increasingly accessible. The real challenge is not generation, but validation. If each strategy is built and tested in isolation, especially through one-off AI-generated code, the research process quickly becomes inconsistent, hard to trust, and inefficient. A reliable framework matters because it provides a stable environment in which ideas can be tested under coherent assumptions and compared meaningfully. Instead of rewriting execution logic, filters, and risk rules for each new idea, the framework separates signal research from trade simulation so that strategies can be evaluated inside a consistent execution environment.

The engine is designed for bar-based research workflows and supports signal inspection, regime-aware strategy routing, execution-aware backtesting, and trade-level post-analysis.

---

## What the framework provides

The framework is organized as a set of research layers that reduce friction between idea generation and reliable evaluation.

At a high level, it helps structure the workflow from:

- market data
- feature construction
- signal generation
- setup selection
- regime conditioning
- execution and trade management
- post-trade analysis

Instead of rebuilding these layers for each new strategy, the user works inside a reusable architecture where the research logic remains explicit and modular.

---
### Framework Architecture Overview

<img width="907" height="469" alt="Screenshot 2026-04-17 at 18 44 44" src="https://github.com/user-attachments/assets/bb59d262-d3d8-451e-ba51-0c7de7b16894" />
<img width="907" height="669" alt="Screenshot 2026-04-17 at 18 39 09" src="https://github.com/user-attachments/assets/b97358bd-aa58-4b6d-802a-921204707eb4" />
<img width="907" height="705" alt="Screenshot 2026-04-17 at 18 39 50" src="https://github.com/user-attachments/assets/981027ca-1c3f-45b3-bdf3-99bcca362833" />
<img width="907" height="72" alt="Screenshot 2026-04-17 at 18 40 21" src="https://github.com/user-attachments/assets/98c073a1-116a-4bce-ab1d-083f4628e795" />

--- 
## Technical layers

### 1. Data layer

The engine starts from standardized OHLCV data and provides a consistent base for multi-timeframe research.  
This includes data loading, alignment, resampling, and synchronization across execution and higher-timeframe context.

This matters because many research errors come from inconsistent indexing, ad hoc resampling, or manual preprocessing repeated across notebooks.

### 2. Feature layer

Features can be defined, reused, compiled, and attached to the research workflow without being tightly coupled to one specific strategy.  
This makes it easier to experiment with indicators, contextual variables, and higher-level market descriptors while keeping the logic traceable.

### 3. Signal layer

A strategy can remain simple: it only needs to express entry logic and produce a signal or a setup-compatible output.  
The framework then handles the rest of the research stack around it.

This separation is important because it prevents execution assumptions from being silently mixed into signal generation.

### 4. Setup layer

Signals can be promoted into named setups and combined inside a multi-setup workflow.  
This allows different entry logics to coexist, be scored, filtered, and routed in a unified structure rather than being tested in isolation.

### 5. Regime layer

The framework supports regime-aware filtering and routing, so setups can be activated, deactivated, or directionally constrained depending on market state.  
This makes it possible to test not only whether a strategy works, but also under which conditions it works.

### 6. Execution layer

The backtest engine handles realistic bar-based execution assumptions, including trade management, session filters, forced-flat logic, costs, and configurable exits.  
This creates a more stable and consistent testing environment than one-off strategy scripts built around a single idea.

Execution costs are modeled explicitly through spread, slippage, and per-lot commission inputs, and are converted into trade-level relative return costs during backtest execution.
See `docs/technical_guide.md` for the execution cost model and conventions.

### 7. Post-analysis layer

Results can be analyzed at the trade, setup, regime, and context level.  
This is a key part of the framework: the goal is not only to generate a backtest, but to make the edge interpretable and easier to diagnose.

---

## Core capabilities

- Standardized OHLCV research workflow
- Multi-timeframe data alignment and context projection
- Reusable feature and indicator integration
- Built-in and user-defined signal generation
- Setup-based signal aggregation and routing
- Regime-aware filtering and directional control
- Configurable exit profiles
- Configurable exit strategies with a Numba-compatible execution bridge
- Realistic CFD execution assumptions:
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
- Fast Numba-based execution

---

## What this reduces in practice

The framework is designed to reduce research friction in a few specific ways:

- less repeated boilerplate across notebooks
- cleaner separation between market logic and execution logic
- easier comparison across strategies under consistent assumptions
- easier reuse of features, setups, and regime logic
- more interpretable results once trades have been executed

In practice, this means the user can spend more time refining hypotheses and less time rebuilding infrastructure around each new test.

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

### Multi-Setup routing and post-trade analysis
<img width="1154" height="576" alt="Screenshot 2026-04-16 at 22 32 34" src="https://github.com/user-attachments/assets/d7074d16-5b90-4f6a-98af-e1ba4275a320" />
<img width="1202" height="384" alt="Screenshot 2026-04-16 at 22 51 02" src="https://github.com/user-attachments/assets/889b381a-fcb4-43b2-8a71-e8d6817a31a1" />

Recommended entry point:

`examples/Framework_Research_Workflow_Demo.ipynb`

---

## Installation

```bash
pip install "git+https://github.com/Arnaud-BARBIER/Multi-strategy-backtest-engine.git@main"
