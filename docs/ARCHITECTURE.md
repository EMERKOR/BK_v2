# Ball Knower v2: Architecture

*TODO: Document after BK-TASK-001 complete*

## Overview

Ball Knower v2 is an NFL prediction and backtesting system.

## Module Structure

```
ball_knower/
├── io/           # Data ingestion and cleaning
├── features/     # Feature engineering
├── datasets/     # Dataset assembly
├── models/       # Prediction models
├── calibration/  # Model calibration
├── backtesting/  # Betting simulation
└── scripts/      # CLI entry points
```

## Data Flow

```
RAW_* -> clean_tables -> game_state -> features -> datasets -> models -> backtesting
```

## Key Files

- `ball_knower/io/clean_tables.py` - Raw to clean transformation
- `ball_knower/features/builder_v2.py` - Feature assembly
- `ball_knower/models/score_model_v2.py` - Score prediction
- `ball_knower/backtesting/engine_v2.py` - Bet simulation

---

*This document will be expanded as the system is verified.*
