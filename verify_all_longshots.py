#!/usr/bin/env python3
"""
LONGSHOT VERIFICATION SCRIPT - Ball Knower v3
Verifies all longshot bets from weeks 4-11 (2025 NFL Season)
against nflverse player statistics.

Usage: python verify_all_longshots.py

Requires: nflverse_player_stats_2025.csv in same directory
(or adjust DATA_PATH below)
"""

import pandas as pd
import sys

# Adjust this path if your CSV is elsewhere
DATA_PATH = 'nflverse_player_stats_2025.csv'

# =============================================================================
# MASTER LONGSHOT LIST - Weeks 4-11
# Format: (week, player_search, stat_col, threshold, odds, prop_desc, reasoning)
# =============================================================================

LONGSHOTS = [
    # =========================================================================
    # WEEK 4
    # =========================================================================
    (4, "Mitchell", "receiving_yards", 30, "+400", "30+ yard catch", 
     "AD Mitchell - downfield threat"),
    (4, "Sinnott", "receiving_tds", 1, "+1500", "TD", 
     "Ben Sinnott - rookie TE upside"),
    (4, "Mariota", "rushing_tds", 1, "+380", "Rush TD", 
     "Mariota - designed QB runs"),
    (4, "Chubb", "rushing_yards", 75, "+300", "75+ rush yds", 
     "Nick Chubb - return from injury"),
    (4, "Chubb", "rushing_yards", 100, "+1020", "100+ rush yds", 
     "Nick Chubb - big game potential"),
    (4, "Miller", "receiving_yards", 3, "-110", "o2.5 rec yds", 
     "Kendre Miller - pass catching role"),
    (4, "Miller", "receiving_yards", 20, "+750", "20+ rec yds", 
     "Kendre Miller - chunk play upside"),
    (4, "Thrash", "receiving_tds", 1, "+1400", "TD", 
     "Jamari Thrash - deep threat"),
    (4, "Whittington", "receiving_tds", 1, "+550", "TD", 
     "Jordan Whittington - red zone role"),
    
    # =========================================================================
    # WEEK 5
    # =========================================================================
    (5, "Lockett", "receiving_tds", 1, "+1000", "TD", 
     "Tyler Lockett - CONFIRMED HIT (fumble recovery)"),
    (5, "Robinson", "receiving_yards", 50, "+300", "50+ rec yds", 
     "Demarcus Robinson - volume upside"),
    (5, "Robinson", "receiving_yards", 75, "+500", "75+ rec yds", 
     "Demarcus Robinson - ceiling play"),
    (5, "Corum", "rushing_yards", 50, "+300", "50+ rush yds", 
     "Blake Corum - workload increase"),
    (5, "Corum", "rushing_tds", 1, "+1600", "last TD", 
     "Blake Corum - goal line role"),
    (5, "Allen", "receiving_tds", 1, "+600", "TD", 
     "Davis Allen - TE red zone"),
    (5, "Farrell", "receiving_tds", 1, "+1500", "TD", 
     "Luke Farrell - blocking TE TD upside"),
    (5, "Davis", "receptions", 3, "+350", "3+ rec", 
     "Isaiah Davis - pass catching RB"),
    (5, "Hyatt", "receiving_yards", 20, "+350", "20+ yard catch", 
     "Hyatt - deep threat"),
    (5, "Ayomanor", "receiving_tds", 1, "+425", "TD", 
     "Elic Ayomanor - red zone looks"),
    
    # =========================================================================
    # WEEK 6
    # =========================================================================
    (6, "Carter", "receptions", 3, "+200", "3+ catches", 
     "Michael Carter - pass catching role"),
    (6, "Bech", "receptions", 3, "+220", "3+ rec", 
     "Jack Bech - slot usage"),
    # Note: Two Washingtons - Parker (WR) and Darnell (TE)
    (6, "Washington", "receiving_yards", 30, "+350", "30+ yards", 
     "Parker Washington - slot WR upside"),
    (6, "Washington", "receiving_tds", 1, "+700", "TD", 
     "Darnell Washington - big TE red zone"),
    (6, "Washington", "receptions", 3, "+350", "3+ rec", 
     "Darnell Washington - target share increase"),
    (6, "Smith", "receptions", 3, "+220", "3+ catches", 
     "Brashad Smith - gadget role"),
    
    # =========================================================================
    # WEEK 7
    # =========================================================================
    (7, "Atwell", "receiving_yards", 30, "+300", "30+ yard catch", 
     "Tutu Atwell - deep speed"),
    (7, "Hunter", "receptions", 6, "+320", "6+ rec", 
     "Travis Hunter - target hog"),
    (7, "Berrios", "receiving_tds", 1, "+2200", "TD", 
     "Braxton Berrios - lottery ticket"),
    (7, "Smith", "receiving_yards", 40, "+550", "40+ yards", 
     "Arian Smith - deep threat"),
    (7, "Washington", "receptions", 4, "+250", "4 rec", 
     "Malik Washington - slot role"),
    (7, "Washington", "receptions", 5, "+600", "5 rec", 
     "Malik Washington - ceiling"),
    (7, "Gordon", "rushing_yards", 30, "+310", "30+ rush yds", 
     "Ollie Gordon - rookie workload"),
    (7, "Gordon", "rushing_yards", 40, "+700", "40+ rush yds", 
     "Ollie Gordon - ceiling"),
    (7, "Miller", "rushing_yards", 20, "+375", "20+ yard rush", 
     "Kendre Miller - explosive run"),
    (7, "Miller", "rushing_tds", 1, "+400", "TD", 
     "Kendre Miller - goal line role"),
    
    # =========================================================================
    # WEEK 8
    # =========================================================================
    (8, "Harris", "receptions", 2, "+225", "2+ rec", 
     "Tre Harris - rookie target share"),
    (8, "Harris", "receptions", 3, "+640", "3+ rec", 
     "Tre Harris - ceiling"),
    (8, "Harris", "receiving_yards", 40, "+750", "40+ yards", 
     "Tre Harris - chunk play"),
    (8, "Odunze", "receiving_yards", 30, "+220", "30+ yard catch", 
     "Rome Odunze - deep target"),
    (8, "Burden", "receiving_yards", 30, "+500", "30+ yard catch", 
     "Luther Burden - explosive play"),
    (8, "Maye", "passing_yards", 300, "+550", "300+ pass yds", 
     "Drake Maye - shootout upside"),
    (8, "Boutte", "receiving_yards", 50, "+250", "50+ rec yds", 
     "Kayshon Boutte - volume increase"),
    (8, "Bond", "receiving_yards", 50, "+350", "50+ rec yds", 
     "Isaiah Bond - deep threat"),
    (8, "Dowdle", "rushing_yards", 100, "+750", "100+ rush yds", 
     "Rico Dowdle - workhorse potential"),
    (8, "Horn", "receiving_yards", 20, "+260", "20+ yds", 
     "Jimme Horn - slot work"),
    
    # =========================================================================
    # WEEK 9 (PREVIOUSLY VERIFIED)
    # =========================================================================
    (9, "Jennings", "rushing_tds", 1, "+400", "TD", 
     "Terrell Jennings - goal-line role (HIT)"),
    (9, "London", "receptions", 8, "+350", "8+ rec", 
     "Drake London - target volume vs weak D (HIT)"),
    (9, "Tuten", "rushing_tds", 1, "+350", "TD", 
     "Bhayshul Tuten - expanded role (HIT)"),
    (9, "Jennings", "rushing_yards", 40, "+320", "40+ rush yds", 
     "Terrell Jennings - rushing volume"),
    (9, "Waddle", "receiving_yards", 100, "+350", "100+ rec yds", 
     "Jaylen Waddle - ceiling play"),
    (9, "Hopkins", "receiving_tds", 1, "+500", "TD", 
     "DeAndre Hopkins - red zone threat"),
    (9, "Mitchell", "rushing_tds", 1, "+750", "TD", 
     "Keaton Mitchell - goal line"),
    (9, "Tuten", "receiving_yards", 20, "+350", "20+ rec yds", 
     "Tuten - pass catching upside"),
    (9, "Tuten", "rushing_yards", 40, "+260", "40+ rush yds", 
     "Tuten - rushing volume"),
    (9, "Tuten", "rushing_yards", 50, "+525", "50+ rush yds", 
     "Tuten - ceiling"),
    (9, "Walker", "rushing_yards", 80, "+320", "80+ rush yds", 
     "Kenneth Walker - volume play"),
    
    # =========================================================================
    # WEEK 10 (PREVIOUSLY VERIFIED)
    # =========================================================================
    (10, "Williams", "receiving_yards", 20, "+225", "20+ yd catch", 
     "Kyle Williams - route share 60.9% post-Boutte (HIT)"),
    (10, "Williams", "receiving_yards", 40, "+375", "40+ rec yds", 
     "Kyle Williams - clear vacancy to fill (HIT)"),
    (10, "Jeanty", "receptions", 4, "+180", "4+ rec", 
     "Ashton Jeanty - receiving role"),
    (10, "Jeanty", "receptions", 5, "+400", "5+ rec", 
     "Ashton Jeanty - ceiling"),
    (10, "Jeanty", "receiving_yards", 40, "+425", "40+ rec yds", 
     "Ashton Jeanty - chunk play"),
    (10, "TeSlaa", "receiving_tds", 1, "+600", "TD", 
     "Isaac TeSlaa - 'positive coachspeak' (LOW quality)"),
    (10, "Dulcich", "receiving_tds", 1, "+600", "TD", 
     "Greg Dulcich - TE red zone"),
    (10, "Dulcich", "receptions", 4, "+280", "4+ rec", 
     "Greg Dulcich - 'Miami loves TEs' speculation"),
    (10, "Dulcich", "receptions", 5, "+560", "5+ rec", 
     "Greg Dulcich - ceiling"),
    (10, "Neal", "rushing_tds", 1, "+650", "TD", 
     "Devin Neal - role projection without data (LOW)"),
    (10, "Jennings", "rushing_tds", 1, "+350", "TD", 
     "Terrell Jennings - prior week success only"),
    
    # =========================================================================
    # WEEK 11 (PREVIOUSLY VERIFIED)
    # =========================================================================
    (11, "Adams", "receiving_yards", 100, "+425", "100+ rec yds", 
     "Davante Adams - bet despite 'injury looming' (LOW quality - 1 yard actual)"),
    (11, "Wright", "rushing_tds", 1, "+1000", "TD", 
     "Jaylen Wright - snap share only (LOW quality)"),
]


def load_data():
    """Load nflverse player stats"""
    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
        print(f"Loaded {len(df)} player-week records from {DATA_PATH}")
        return df
    except FileNotFoundError:
        print(f"ERROR: Could not find {DATA_PATH}")
        print("Please ensure the CSV is in the current directory or adjust DATA_PATH")
        sys.exit(1)


def verify_longshot(df, week, name_search, stat_col, threshold, odds, desc, reasoning):
    """Verify a single longshot bet against the data"""
    week_data = df[df['week'] == week]
    
    # Search for player
    player_rows = week_data[
        week_data['player_display_name'].str.contains(name_search, case=False, na=False)
    ]
    
    if len(player_rows) == 0:
        return {
            'week': week,
            'player': name_search,
            'prop': desc,
            'odds': odds,
            'actual': 'NOT FOUND',
            'threshold': threshold,
            'result': 'UNKNOWN',
            'reasoning': reasoning
        }
    
    # Take first match
    row = player_rows.iloc[0]
    player_name = row['player_display_name']
    
    # Handle TD props (need to check both rushing and receiving TDs)
    if stat_col == 'rushing_tds':
        actual = row.get('rushing_tds', 0)
        if pd.isna(actual):
            actual = 0
        # Also check receiving TDs for RBs
        rec_tds = row.get('receiving_tds', 0)
        if pd.isna(rec_tds):
            rec_tds = 0
        total_tds = actual + rec_tds
        hit = total_tds >= 1
    elif stat_col == 'receiving_tds':
        actual = row.get('receiving_tds', 0)
        if pd.isna(actual):
            actual = 0
        hit = actual >= 1
    elif stat_col == 'passing_yards':
        actual = row.get('passing_yards', 0)
        if pd.isna(actual):
            actual = 0
        hit = actual >= threshold
    elif stat_col in df.columns:
        actual = row.get(stat_col, 0)
        if pd.isna(actual):
            actual = 0
        hit = actual >= threshold
    else:
        return {
            'week': week,
            'player': player_name,
            'prop': desc,
            'odds': odds,
            'actual': f'COL {stat_col} NOT FOUND',
            'threshold': threshold,
            'result': 'UNKNOWN',
            'reasoning': reasoning
        }
    
    return {
        'week': week,
        'player': player_name,
        'prop': desc,
        'odds': odds,
        'actual': actual,
        'threshold': threshold,
        'result': 'HIT' if hit else 'MISS',
        'reasoning': reasoning
    }


def calculate_pl(results, bet_size=50):
    """Calculate profit/loss assuming flat bet size"""
    total_risked = 0
    total_return = 0
    
    for r in results:
        if r['result'] == 'UNKNOWN':
            continue
            
        total_risked += bet_size
        
        if r['result'] == 'HIT':
            odds_str = r['odds'].replace('+', '').replace('-', '')
            try:
                odds_val = int(odds_str)
                if r['odds'].startswith('+'):
                    profit = bet_size * (odds_val / 100)
                else:
                    profit = bet_size * (100 / odds_val)
                total_return += bet_size + profit
            except ValueError:
                pass
    
    return total_risked, total_return


def main():
    print("=" * 90)
    print("LONGSHOT VERIFICATION - 2025 NFL SEASON (WEEKS 4-11)")
    print("=" * 90)
    
    df = load_data()
    results = []
    
    # Verify each longshot
    for longshot in LONGSHOTS:
        week, name_search, stat_col, threshold, odds, desc, reasoning = longshot
        result = verify_longshot(df, week, name_search, stat_col, threshold, odds, desc, reasoning)
        results.append(result)
    
    # Print results by week
    current_week = 0
    for r in results:
        if r['week'] != current_week:
            current_week = r['week']
            print(f"\n{'='*90}")
            print(f"WEEK {current_week}")
            print(f"{'='*90}")
            print(f"{'Player':<25} {'Prop':<20} {'Odds':<8} {'Actual':<10} {'Need':<8} {'Result':<8}")
            print("-" * 90)
        
        status = "HIT" if r['result'] == 'HIT' else "MISS" if r['result'] == 'MISS' else "???"
        print(f"{r['player']:<25} {r['prop']:<20} {r['odds']:<8} {str(r['actual']):<10} {r['threshold']:<8} {status:<8}")
    
    # Summary statistics
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    
    hits = [r for r in results if r['result'] == 'HIT']
    misses = [r for r in results if r['result'] == 'MISS']
    unknown = [r for r in results if r['result'] == 'UNKNOWN']
    
    print(f"\nTotal Longshots: {len(results)}")
    print(f"Verified: {len(hits) + len(misses)}")
    print(f"Unknown: {len(unknown)}")
    print(f"\nRecord: {len(hits)}-{len(misses)}")
    if (len(hits) + len(misses)) > 0:
        print(f"Hit Rate: {len(hits)/(len(hits)+len(misses))*100:.1f}%")
    
    # P/L calculation
    total_risked, total_return = calculate_pl(results)
    
    print("\n" + "=" * 90)
    print("P/L BREAKDOWN (assuming $50/bet)")
    print("=" * 90)
    
    print("\nHITS:")
    for r in hits:
        odds_str = r['odds'].replace('+', '')
        try:
            odds_val = int(odds_str)
            profit = 50 * (odds_val / 100)
            print(f"  + {r['player']:<20} {r['prop']:<20} {r['odds']:<8} +${profit:.2f}")
        except ValueError:
            print(f"  + {r['player']:<20} {r['prop']:<20} {r['odds']:<8} (odds parse error)")
    
    print("\nMISSES:")
    for r in misses:
        print(f"  - {r['player']:<20} {r['prop']:<20} {r['odds']:<8} -$50.00")
    
    if total_risked > 0:
        print(f"\nTotal Risked: ${total_risked:.2f}")
        print(f"Total Return: ${total_return:.2f}")
        print(f"Net P/L: ${total_return - total_risked:.2f}")
        print(f"ROI: {((total_return - total_risked) / total_risked) * 100:.1f}%")
    
    # Week-by-week breakdown
    print("\n" + "=" * 90)
    print("WEEK-BY-WEEK BREAKDOWN")
    print("=" * 90)
    
    for week in range(4, 12):
        week_results = [r for r in results if r['week'] == week]
        week_hits = [r for r in week_results if r['result'] == 'HIT']
        week_misses = [r for r in week_results if r['result'] == 'MISS']
        
        if len(week_hits) + len(week_misses) > 0:
            week_risked, week_return = calculate_pl(week_results)
            week_pl = week_return - week_risked
            week_roi = (week_pl / week_risked * 100) if week_risked > 0 else 0
            print(f"Week {week:2}: {len(week_hits):2}-{len(week_misses):2} | P/L: ${week_pl:+7.2f} | ROI: {week_roi:+6.1f}%")
    
    # Reasoning quality analysis
    print("\n" + "=" * 90)
    print("REASONING QUALITY NOTES")
    print("=" * 90)
    
    print("\nCONFIRMED HITS (analyze methodology):")
    for r in hits:
        print(f"\n  {r['player']} {r['prop']} {r['odds']}")
        print(f"    Reasoning: {r['reasoning']}")
    
    print("\n\nNOTABLE MISSES (methodology failures):")
    notable_misses = [r for r in misses if 'LOW' in r['reasoning'] or 'coachspeak' in r['reasoning'].lower()]
    for r in notable_misses:
        print(f"\n  {r['player']} {r['prop']} {r['odds']}")
        print(f"    Reasoning: {r['reasoning']}")
        print(f"    Actual: {r['actual']}")


if __name__ == "__main__":
    main()
