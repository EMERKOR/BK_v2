# Ball Knower v2 Runbook

## 1. Start of Thread
```bash
cd /workspaces/BK_v2
bash scripts/thread_diagnostic.sh
```

## 2. Build Features
```bash
python -c "from ball_knower.features.builder_v2 import build_features_v2; df = build_features_v2(2024, 10); print(f'{len(df)} games, {len(df.columns)} cols')"
```

## 3. Build Test Games
```bash
python ball_knower/scripts/build_test_games_v2.py --season 2024
```

## 4. Run Backtest
```bash
python ball_knower/scripts/run_backtest_v2.py --config configs/backtest_default.json --test-games data/test_games/test_games_2024.parquet
```

## 5. Git Workflow
```bash
git add -A && git commit -m "msg" && git push
```

## Week Reference
1-18 regular, 19 Wild Card, 20 Divisional, 21 Conference, 22 Super Bowl
