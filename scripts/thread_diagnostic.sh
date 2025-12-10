#!/bin/bash
echo "==========================================="
echo "BALL KNOWER THREAD DIAGNOSTIC"
echo "==========================================="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

echo "=== GIT STATE ==="
git log --oneline -5
echo ""
git status --short
if [ -z "$(git status --short)" ]; then
    echo "(working tree clean)"
fi
echo ""

echo "=== CURRENT BRANCH ==="
git branch --show-current
echo ""

echo "=== ROADMAP TASK STATUS ==="
grep -E "^\*\*Status:\*\*" ROADMAP.md 2>/dev/null | head -15 || echo "ROADMAP.md not found"
echo ""

echo "=== DATA INVENTORY ==="
echo "Coverage offense files: $(ls data/RAW_fantasypoints/coverage/offense/ 2>/dev/null | wc -l)"
echo "Coverage defense files: $(ls data/RAW_fantasypoints/coverage/defense/ 2>/dev/null | wc -l)"
echo "PBP files: $(ls data/RAW_pbp/ 2>/dev/null | wc -l)"
echo "Schedule seasons: $(ls data/RAW_schedule/ 2>/dev/null | wc -l)"
echo "Market spread seasons: $(ls data/RAW_market/spread/ 2>/dev/null | wc -l)"
echo ""

echo "=== COVERAGE FILES BY SEASON ==="
echo "Offense:"
ls data/RAW_fantasypoints/coverage/offense/ 2>/dev/null | cut -d'_' -f3 | sort | uniq -c || echo "  None"
echo "Defense:"
ls data/RAW_fantasypoints/coverage/defense/ 2>/dev/null | cut -d'_' -f3 | sort | uniq -c || echo "  None"
echo ""

echo "=== RECENT WORK (last 3 commits) ==="
git log --oneline -3 --format="%h %s (%cr)"
echo ""

echo "=== WORKLOG LAST SESSION ==="
if [ -f "WORKLOG.md" ]; then
    # Find line numbers of all session headers, excluding template placeholder
    SESSION_LINES=$(grep -n "^## Session: 2" WORKLOG.md | tail -1 | cut -d: -f1)
    if [ -n "$SESSION_LINES" ]; then
        # Get 25 lines starting from last real session
        sed -n "${SESSION_LINES},\$p" WORKLOG.md | head -25
    else
        echo "No completed sessions found in WORKLOG.md"
    fi
else
    echo "WORKLOG.md not found"
fi
echo ""
echo "==========================================="
echo "END DIAGNOSTIC"
echo "==========================================="
