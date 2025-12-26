#!/usr/bin/env python3
"""
Patch script to fix CLV calculation in run_backtest_v2.py

Issue: CLV was measuring model prediction error, not betting value.
Fix: CLV now measures cover margin from the bet's perspective.

Usage:
    python ball_knower/scripts/patch_clv_fix.py
"""
from pathlib import Path

def main():
    target = Path("ball_knower/scripts/run_backtest_v2.py")
    
    if not target.exists():
        print(f"ERROR: {target} not found")
        return 1
    
    content = target.read_text()
    
    # The old (incorrect) CLV calculation
    old_code = '''    # CLV: How much our predicted spread differed from actual result
    # Positive CLV = we were on the right side of the "true" line
    df["clv"] = np.where(
        df["bet_side"] == "home",
        df["spread_actual"] - df["spread_pred"],  # Home bet: actual > pred is good
        df["spread_pred"] - df["spread_actual"]   # Away bet: pred > actual is good
    )'''
    
    # The new (correct) CLV calculation
    new_code = '''    # CLV: How much we beat the spread by from our bet's perspective
    # Positive CLV = we covered, negative CLV = we didn't cover
    # This is the cover margin from the perspective of the side we bet
    df["clv"] = np.where(
        df["bet_side"] == "home",
        df["cover_margin"],      # positive = home covered
        -df["cover_margin"]      # positive = away covered
    )'''
    
    if old_code not in content:
        # Check if already patched
        if "cover_margin" in content and 'df["clv"]' in content:
            print("File appears to already be patched (CLV uses cover_margin)")
            return 0
        print("ERROR: Could not find the expected CLV code block to replace")
        print("Manual inspection required")
        return 1
    
    new_content = content.replace(old_code, new_code)
    
    if new_content == content:
        print("ERROR: Replacement had no effect")
        return 1
    
    target.write_text(new_content)
    print(f"SUCCESS: Patched {target}")
    print("\nChange made:")
    print("  OLD: CLV = spread_actual - spread_pred (model error)")
    print("  NEW: CLV = cover_margin from bet perspective (actual betting value)")
    print("\nRe-run backtest to see corrected CLV metrics:")
    print("  python ball_knower/scripts/run_backtest_v2.py --year 2024")
    
    return 0

if __name__ == "__main__":
    exit(main())
