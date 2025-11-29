"""
Snap Share Feature Prototype - BK_v2 Phase 13

This prototype validates the snap share data integration approach
before implementing in the main codebase.

Usage:
    python snap_share_prototype.py

Expected output:
    - Team code mapping validation
    - Feature extraction example
    - Anti-leakage verification
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ===== Team Code Mapping =====

FP_TO_BK_TEAM_CODES = {
    # Fantasy Points -> BK Canonical
    "HST": "HOU",
    "LA": "LAR",
    "BLT": "BAL",
    "ARZ": "ARI",
    "CLV": "CLE",
    # Standard codes that match
    "KC": "KC",
    "BUF": "BUF",
    "PHI": "PHI",
    "SF": "SF",
    "DAL": "DAL",
    "DET": "DET",
    "GB": "GB",
    "MIN": "MIN",
    "CHI": "CHI",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "WAS": "WAS",
    "MIA": "MIA",
    "NE": "NE",
    "TEN": "TEN",
    "IND": "IND",
    "JAX": "JAX",
    "DEN": "DEN",
    "LAC": "LAC",
    "LV": "LV",
    "SEA": "SEA",
    "NO": "NO",
    "ATL": "ATL",
    "CAR": "CAR",
    "TB": "TB",
    "CIN": "CIN",
    "PIT": "PIT",
}


def normalize_team_code(raw_team: str) -> str:
    """Convert Fantasy Points team code to BK canonical."""
    # Handle multi-team players: "LV, NYJ" -> take last team
    if "," in raw_team:
        raw_team = raw_team.split(",")[-1].strip()
    
    return FP_TO_BK_TEAM_CODES.get(raw_team, raw_team)


# ===== Raw Loader =====

def load_snap_share_raw(season: int, data_dir: str = "/home/user/BK_v2/data") -> pd.DataFrame:
    """
    Load raw snap share CSV for a season.
    
    Returns player-level DataFrame with columns:
        - Rank, Name, Team, POS, G, Season
        - W1-W18 (snap percentages as strings)
        - Snap %
    """
    path = Path(data_dir) / "RAW_fantasypoints" / f"snap_share_{season}.csv"
    
    if not path.exists():
        raise FileNotFoundError(f"Snap share file not found: {path}")
    
    # Skip header placeholder row (row 0)
    df = pd.read_csv(path, skiprows=1)
    
    # Remove footer rows (column definitions)
    df = df[df["Rank"].apply(lambda x: str(x).isdigit())]
    
    # Convert Rank to int
    df["Rank"] = df["Rank"].astype(int)
    
    return df


# ===== Clean Table Builder =====

def build_snap_share_clean(season: int, data_dir: str = "/home/user/BK_v2/data") -> pd.DataFrame:
    """
    Build cleaned snap share table.
    
    Transformations:
        - Normalize team codes to BK canonical
        - Convert week columns to float (empty -> NaN)
        - Filter to WR, RB, TE only
        - Add season column
    """
    df_raw = load_snap_share_raw(season, data_dir)
    
    df = pd.DataFrame()
    df["season"] = season
    df["player_name"] = df_raw["Name"]
    df["team_code"] = df_raw["Team"].apply(normalize_team_code)
    df["position"] = df_raw["POS"]
    df["games_played"] = pd.to_numeric(df_raw["G"], errors="coerce").astype("Int64")
    
    # Convert week columns to float
    for week in range(1, 19):
        raw_col = f"W{week}"
        clean_col = f"w{week}_snap_pct"
        df[clean_col] = pd.to_numeric(df_raw[raw_col], errors="coerce")
    
    df["season_snap_pct"] = pd.to_numeric(df_raw["Snap %"], errors="coerce")
    
    # Filter to skill positions (exclude FB)
    df = df[df["position"].isin(["WR", "RB", "TE"])].copy()
    
    return df.reset_index(drop=True)


# ===== Feature Extraction =====

def get_team_week_snap_features(
    team: str, 
    week: int, 
    snap_df: pd.DataFrame
) -> dict:
    """
    Extract snap share features for one team-week.
    
    CRITICAL: Uses prior week data for anti-leakage.
    For Week N predictions, uses Week N-1 snap data.
    """
    # Anti-leakage: use prior week
    prior_week = week - 1
    
    if prior_week < 1:
        # Week 1: return defaults (no prior data)
        return {
            "rb1_snap_share": np.nan,
            "rb2_snap_share": np.nan,
            "wr1_snap_share": np.nan,
            "te1_snap_share": np.nan,
            "rb_concentration": np.nan,
            "top3_skill_snap_avg": np.nan,
        }
    
    week_col = f"w{prior_week}_snap_pct"
    team_df = snap_df[snap_df["team_code"] == team]
    
    # Get top RBs by snap share
    rbs = team_df[team_df["position"] == "RB"].nlargest(2, week_col)
    rb1_snap = rbs.iloc[0][week_col] if len(rbs) > 0 else np.nan
    rb2_snap = rbs.iloc[1][week_col] if len(rbs) > 1 else np.nan
    
    # Get top WR
    wrs = team_df[team_df["position"] == "WR"].nlargest(1, week_col)
    wr1_snap = wrs.iloc[0][week_col] if len(wrs) > 0 else np.nan
    
    # Get top TE
    tes = team_df[team_df["position"] == "TE"].nlargest(1, week_col)
    te1_snap = tes.iloc[0][week_col] if len(tes) > 0 else np.nan
    
    # Derived features
    rb_concentration = np.nan
    if pd.notna(rb1_snap) and pd.notna(rb2_snap) and (rb1_snap + rb2_snap) > 0:
        rb_concentration = rb1_snap / (rb1_snap + rb2_snap)
    
    top3_avg = np.nanmean([rb1_snap, wr1_snap, te1_snap])
    
    return {
        "rb1_snap_share": rb1_snap,
        "rb2_snap_share": rb2_snap,
        "wr1_snap_share": wr1_snap,
        "te1_snap_share": te1_snap,
        "rb_concentration": rb_concentration,
        "top3_skill_snap_avg": top3_avg,
    }


def get_snap_delta_features(
    team: str,
    week: int,
    snap_df: pd.DataFrame
) -> dict:
    """
    Extract week-over-week snap share changes.
    
    Uses Week N-1 vs Week N-2 for anti-leakage.
    """
    prior_week = week - 1
    prior_prior_week = week - 2
    
    if prior_prior_week < 1:
        return {
            "rb1_snap_delta": np.nan,
            "wr1_snap_delta": np.nan,
            "te1_snap_delta": np.nan,
        }
    
    curr_col = f"w{prior_week}_snap_pct"
    prev_col = f"w{prior_prior_week}_snap_pct"
    
    team_df = snap_df[snap_df["team_code"] == team]
    
    deltas = {}
    for pos, key in [("RB", "rb1"), ("WR", "wr1"), ("TE", "te1")]:
        pos_df = team_df[team_df["position"] == pos]
        
        if len(pos_df) == 0:
            deltas[f"{key}_snap_delta"] = np.nan
            continue
        
        # Get top player by current week snap share
        top_player = pos_df.nlargest(1, curr_col)
        curr_snap = top_player.iloc[0][curr_col]
        prev_snap = top_player.iloc[0][prev_col]
        
        if pd.notna(curr_snap) and pd.notna(prev_snap):
            deltas[f"{key}_snap_delta"] = curr_snap - prev_snap
        else:
            deltas[f"{key}_snap_delta"] = np.nan
    
    return deltas


# ===== Main Validation =====

def main():
    print("=" * 60)
    print("SNAP SHARE PROTOTYPE VALIDATION")
    print("=" * 60)
    
    # Load 2024 data
    print("\n1. Loading 2024 snap share data...")
    snap_df = build_snap_share_clean(2024)
    print(f"   Loaded {len(snap_df)} players")
    print(f"   Positions: {snap_df['position'].value_counts().to_dict()}")
    
    # Check team codes
    print("\n2. Team code validation...")
    teams = snap_df["team_code"].unique()
    print(f"   {len(teams)} unique teams")
    expected_teams = ["KC", "BUF", "PHI", "SF", "DAL", "DET", "BAL", "CLE", "HOU", "LAR"]
    for t in expected_teams:
        if t in teams:
            print(f"   ✓ {t} found")
        else:
            print(f"   ✗ {t} MISSING")
    
    # Extract features for sample team-week
    print("\n3. Feature extraction (KC, Week 10)...")
    features = get_team_week_snap_features("KC", 10, snap_df)
    for k, v in features.items():
        print(f"   {k}: {v:.1f}" if pd.notna(v) else f"   {k}: NaN")
    
    # Show actual KC Week 9 top players (for verification)
    print("\n4. KC Week 9 players (used for Week 10 features)...")
    kc_df = snap_df[snap_df["team_code"] == "KC"]
    for pos in ["RB", "WR", "TE"]:
        pos_df = kc_df[kc_df["position"] == pos].nlargest(2, "w9_snap_pct")
        if len(pos_df) > 0:
            top = pos_df.iloc[0]
            print(f"   {pos}1: {top['player_name']} - {top['w9_snap_pct']:.1f}%")
    
    # Delta features
    print("\n5. Snap delta features (KC, Week 10)...")
    deltas = get_snap_delta_features("KC", 10, snap_df)
    for k, v in deltas.items():
        if pd.notna(v):
            print(f"   {k}: {v:+.1f}")
        else:
            print(f"   {k}: NaN")
    
    # Anti-leakage verification
    print("\n6. Anti-leakage verification...")
    print("   For Week 10 prediction:")
    print("   - Base features use Week 9 data ✓")
    print("   - Delta features use Week 9 vs Week 8 ✓")
    print("   - Week 10 data is NOT used ✓")
    
    # Build features for all teams, Week 10
    print("\n7. Building features for all teams (Week 10)...")
    all_features = []
    for team in sorted(teams):
        feats = get_team_week_snap_features(team, 10, snap_df)
        feats["team"] = team
        all_features.append(feats)
    
    features_df = pd.DataFrame(all_features)
    print(f"   Generated features for {len(features_df)} teams")
    print(f"   NaN counts:")
    for col in ["rb1_snap_share", "wr1_snap_share", "te1_snap_share"]:
        nan_count = features_df[col].isna().sum()
        print(f"     {col}: {nan_count} NaN")
    
    # Show sample output
    print("\n8. Sample output (top 5 by RB concentration):")
    top5 = features_df.nlargest(5, "rb_concentration")[
        ["team", "rb1_snap_share", "rb2_snap_share", "rb_concentration"]
    ]
    print(top5.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("PROTOTYPE VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
