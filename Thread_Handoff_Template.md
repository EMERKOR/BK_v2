# Ball Knower - Thread Handoff Protocol

## Purpose

At the end of each thread, Claude fills out a handoff document. User copies it into the first message of the next thread. This ensures continuity and prevents context loss between threads.

---

## Part 1: Static Context (Always Applies)

### Project Overview

**Ball Knower** is an NFL betting prediction system that approximates sharp book pricing across spreads, totals, moneylines, and player props.

**Repository:** https://github.com/EMERKOR/BK_v2 (public)  
**Environment:** GitHub Codespaces  
**Primary Docs:**
- `ROADMAP.md` — task breakdown and status tracking
- `WORKLOG.md` — session history, decisions, key content from files not in project knowledge
- `NFL_markets_analysis.md` — research foundation (in project knowledge)
- `scripts/thread_diagnostic.sh` — run at start of every thread

### User Context

- User does not write code — provide copy/paste ready commands for Codespaces terminal
- User knows football and betting markets well
- User provides subjective observations; Claude provides technical implementation
- Explain technical concepts in plain terms
- User will challenge assumptions — this is collaborative, not delegated

### Execution Model

**CRITICAL: Understand how work gets done**

1. User has the repo open in GitHub Codespaces
2. Claude provides terminal commands
3. User runs commands in Codespaces terminal
4. User pastes output back to Claude
5. Claude analyzes output and provides next steps

**Claude should NEVER:**
- Try to clone the repo or access files directly without user running commands
- Assume it can see file contents without user providing them or without checking project knowledge
- Proceed with implementation without seeing actual code first
- Make changes without explicit approval
- Make factual claims without citing source

**Claude should ALWAYS:**
- Start threads by having user run `bash scripts/thread_diagnostic.sh`
- Produce a Pre-Flight Checklist before proposing any solutions
- Emit `📝 LOG:` markers for significant discoveries during conversation
- Tag factual claims with source: `[CODE]`, `[RESEARCH]`, `[WORKLOG]`, `[VERIFIED]`, or `[ASSUMPTION - NEEDS VERIFICATION]`
- Wait for user confirmation before implementing

### Critical Rules

1. **No assumptions** — ask before proceeding if anything is unclear
2. **No unauthorized changes** — propose first, implement after approval
3. **Cite sources** — every factual claim needs a source tag
4. **Verify before acting** — read code/docs before modifying
5. **One thread per phase** — complete all tasks, validate, update docs, then new thread
6. **Test before declaring done** — provide validation commands, confirm they pass
7. **No loose ends** — fix issues now, not later

### Technical Reference

**Spread Convention:**
- Negative = home team favored (e.g., -3 means home favored by 3)
- `spread_edge = predicted_spread + market_closing_spread`
- Positive edge = bet home, negative edge = bet away

**Team Code Normalization:**
- Always use: LAR (not LA), LV (not OAK), LAC (not SD)
- Normalization function: `normalize_team_code(code, "nflverse")`

**Data Locations:**
- Raw data: `data/RAW_*/`
- Clean data: `data/clean/` (gitignored, rebuilds from RAW)
- Datasets: `data/datasets/` (gitignored, rebuilds from clean)
- Coverage: `data/RAW_fantasypoints/coverage/{offense|defense}/`
- Market spreads: `data/clean/market_lines_spread_clean/` (gitignored)
- Predictions: `data/predictions/score_model_v2/` (gitignored)

**Operational Context:**
- Working directory: `/workspaces/BK_v2`
- All commands assume this as cwd
- File creation tools must use absolute paths starting with `/workspaces/BK_v2/`
- Gitignored dirs (clean/, datasets/, predictions/) may be empty after codespace restart

**Model Targets (from research):**
- 53-56% ATS accuracy
- 4-8% ROI
- Positive CLV as leading indicator

---

## Part 2: Mandatory Pre-Flight Checklist

**Claude MUST produce this as first response in every thread, BEFORE proposing any solutions:**
````markdown
## Pre-Flight Checklist

### 1. Diagnostic Output
[User runs `bash scripts/thread_diagnostic.sh`, pastes output here]

### 2. Repo State Confirmed
- Current commit: [hash]
- Working tree: [clean/dirty]
- Branch: [should be main]

### 3. Decisions Already Made (from WORKLOG.md)
[List relevant decisions — DO NOT REVISIT these]
- 
- 

### 4. Key Context for This Task (from WORKLOG.md)
[List relevant facts, code locations, file content excerpts]
-
-

### 5. What I Need Before Proceeding
[List files to read, questions to answer]
-
-

### 6. What I Will NOT Do Without Approval
- Implement any code changes
- Modify architectural decisions
- Make assumptions about data or logic
````

**User confirms checklist is correct, THEN Claude proceeds to propose solutions.**

---

## Part 3: Inline Logging

During conversation, Claude emits markers for significant facts:
````
📝 LOG: load_fp_coverage_matrix_raw() already exists with correct path at line 127
📝 LOG: COVERAGE_MATRIX_ANALYSIS.md specifies BOTH views needed
📝 LOG: Decision — fix loader to call existing function, don't rename
````

At thread end, these are compiled into WORKLOG.md.

---

## Part 4: End of Thread Protocol

**Claude completes this before generating handoff:**

- [ ] All working code committed with descriptive message
- [ ] Changes pushed to origin/main
- [ ] ROADMAP.md task status updated
- [ ] WORKLOG.md updated with session log
- [ ] Any new issues added to relevant docs
- [ ] Verified with `bash scripts/thread_diagnostic.sh`
- [ ] Asked user "Anything to add before handoff?"

---

## Part 5: Handoff Template

**Claude fills this out at end of each thread:**
````markdown
## Thread Handoff — [DATE]

**Completed Thread:** [Thread Name]
**Next Thread:** [Next Thread Name]

---

### Roadmap Position

**Completed Task(s):** [What was just finished]
**Next Task:** [What comes next per ROADMAP.md]

---

### What Was Done

[Bullet list of concrete accomplishments]
- 
- 

### Commits Made

| Hash | Message |
|------|---------|
| [hash] | [message] |

### Files Modified

| File | Change |
|------|--------|
| [filepath] | [what changed] |

---

### Repo State at Handoff

**Current Commit:** [hash]
**Working Tree:** clean
**Branch:** main

---

### Key Context for Next Thread

**Decisions made (DO NOT REVISIT):**
- [decision] — [rationale] — [source]

**Code locations relevant to next task:**
- [file] — [what it contains]

**Key content from files not in project knowledge:**
[Excerpt critical content so next thread has it]

---

### User Notes

[Anything user wants to add]

---

## First Commands for Next Thread
```bash
cd /workspaces/BK_v2
bash scripts/thread_diagnostic.sh
```

Paste output, then Claude produces Pre-Flight Checklist before proposing any solutions.
````
