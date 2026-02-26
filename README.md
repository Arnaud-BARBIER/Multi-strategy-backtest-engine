# Multi-strategy-backtest-engine
Object-oriented backtesting framework featuring pluggable signal modules, configurable execution logic and state-managed trade lifecycle.

<img width="1108" height="693" alt="Screenshot 2026-02-26 at 22 24 52" src="https://github.com/user-attachments/assets/4630ed94-a429-454d-9ab7-455e5f808cfd" />


## Project Overview 

### Motivation

This project was built to move from ad-hoc procedural backtesting
to a modular, scalable architecture suitable for serious research.

Key objectives:
- Decouple data pipeline, strategy logic, and execution engine
- Support multi-entry and stateful position tracking
- Implement realistic execution mechanics (BE, trailing, reverse)
- Prepare the ground for parameter optimization and multi-strategy testing


### Architecture Overview

- **DataPipeline**
  - Feature engineering (EMA, ATR, etc.)

- **Strategy Layer**
  - Entry signal generation

- **BacktestEngine**
  - Stateful execution logic
  - Risk management
  - Exit handling

- **Trade Analytics**
  - Performance metrics
  - MAE / MFE tracking



### Signal Architecture

Clear separation between signal generation and conditional filtering logic.

Strategies remain independent from execution constraints,
while a unified filtering layer enables systematic refinement
and optimised edge development across different models.


### Main Core Features

- Stateful multi-position engine
- Break-even logic with delayed activation
- ATR-based trailing stop with trigger price activation
- Session-based entry limiter
- Reverse mode support
- Gap filtering
- Observation metrics (MAE / MFE / hold analysis)
- Optimized NumPy-based iteration for moderate scalability

### Known Limitations

- No slippage / commission modeling yet
- No portfolio-level allocation layer
- Lack of formal out-of-sample validation, robustness testing,
and statistical significance assessment.


### Future Work

- Strategy injection abstraction layer
- Portfolio allocation module
- High speed backtest engine for grid optimisaton
- Parameter grid optimization
- Slippage and transaction cost modeling
- Development of a robust statistical validation layer


### Future Projects

I Anticipate from statistical analysis to highlight
that optimal parameter configurations vary significantly
across different market regimes especially with an EMA entr signal strategy.

To address this structural instability, the next development phase
will focus on building a regime detection engine using XGBoost
to classify market states and dynamically adapt strategy parameters.



## Engine R&D process 






  
