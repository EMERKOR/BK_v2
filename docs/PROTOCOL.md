# Ball Knower v2: Operating Protocol

*Version 2.1 — December 3, 2025*
*Revised with GPT feedback*

---

## Section 1: Roles and Responsibilities

### Project Lead (Emerson)

- Domain authority on NFL, betting markets, and project goals
- Runs Codespaces terminal — all verification happens through you
- Delivers instructions to implementation agents (Claude Code, Codex)
- Gathers research and external data
- **Only person who may merge to main**
- Final approval on all merges and deployments
- Decides project direction and priorities

### Implementation Agent (Claude Code / Codex)

- Writes code per specifications
- Executes only what is specified — no "improvements" or additions
- Reports exactly what was done, what succeeded, what failed
- Asks for clarification when instructions are ambiguous
- Never makes architectural decisions independently
- **No prior conversation context is available. Every session begins with only the materials supplied in the prompt.**
- **Implementation uses Claude Code / Codex attached directly to the GitHub repo. All planning and auditing must go through the Execution Layer (Section 11).**
- First message of every session must include the reset statement and ticket ID

### Planning Agent (Claude API)

- Produces specifications for implementation
- Reviews implementation results
- Maintains documentation
- All claims must be cited (see Section 7)
- **No prior conversation context is available. Every session begins with only the materials supplied in the prompt.**
- Never references previous chat threads

### Outside Auditor (GPT via API)

- **Must be a different model family than Planning Agent**
- **Auditor is never run in a chat UI — only via llm_api.py**
- Reviews work with zero prior context
- Verifies claims against raw evidence only
- Has no access to any project knowledge base or chat history
- Can challenge any assumption
- Signs off before work is considered complete
- **No prior conversation context is available. Every session begins with only the materials supplied in the prompt.**

### Execution Layer (scripts/llm_api.py)

- Deterministic Python script that makes all LLM API calls
- Ensures only whitelisted context is sent
- Logs all inputs and outputs to JSON
- Prevents chat drift completely
- See Section 11 for implementation

---

## Section 2: Order of Operations

Every task follows this sequence. No step is skipped.

### Task Identification

Every task must have a ticket ID:

```
BK-TASK-###: short-description
```

All commits, specs, logs, audits, and tags must reference this ID.

### Phase 1: Specification

1. Assign ticket ID (next sequential number)
2. Define the task in plain language
3. State preconditions (what must be true before starting):
   - Required files that must exist
   - Required version tags that must be present
   - Required tests that must pass
   - Required data files that must exist
4. State success criteria (how we know it worked)
5. List files that will be modified
6. Planning Agent produces implementation spec

**Gate:** Project Lead approves spec before proceeding.

### Phase 2: Precondition Verification

1. Project Lead runs diagnostic commands in Codespaces
2. Output is shared with Planning Agent
3. Planning Agent confirms each precondition is met with citation to evidence
4. If any precondition fails, STOP — do not proceed to implementation

**Gate:** All preconditions verified with terminal output, each cited.

### Phase 3: Checkpoint

1. Tag current repo state: `git tag pre-BK-TASK-###-YYYYMMDD`
2. Confirm tag exists: `git tag -l | grep BK-TASK-###`
3. Record tag in task log

**Gate:** Tag confirmed before any code changes.

### Phase 4: Implementation

1. Project Lead delivers spec to Implementation Agent
2. Implementation Agent executes exactly as specified
3. Implementation Agent reports what was done
4. Project Lead commits changes with descriptive message

**Gate:** Changes committed to feature branch.

### Phase 5: Verification

1. Project Lead runs test suite: `pytest --tb=short`
2. Project Lead runs any task-specific verification commands
3. All output shared with Planning Agent
4. Planning Agent confirms success criteria are met with citation to evidence

**Gate:** All tests pass, success criteria verified with evidence.

### Phase 6: Audit

1. Project Lead shares task summary and evidence with Outside Auditor
2. Auditor reviews with fresh context
3. Auditor challenges any unverified claims
4. Auditor signs off or requests remediation

**Gate:** Auditor sign-off received.

### Phase 7: Merge and Document

1. Merge feature branch to main
2. Delete feature branch
3. Update relevant documentation
4. Record task completion in project log

**Gate:** Documentation updated, branch cleaned up.

---

## Section 3: Repo Hygiene

### Branch Rules

- `main` is the single source of truth
- **Only the Project Lead may merge to main**
- Feature branches are named: `feature/BK-TASK-###-short-name`
- Feature branches should be short-lived (target 48 hours or less)
- Branches older than 48 hours must either be merged, closed, or explicitly extended in PROJECT_LOG.md with justification
- No direct commits to `main` — all changes via feature branch + merge
- Never delete branches with unmerged commits without explicit backup

### Security Rules

- `.env` and any file containing API keys must be in `.gitignore`
- Never commit secrets to the repository

### Commit Rules

- Every commit has a descriptive message
- Format: `[BK-TASK-###][component] action taken`
- **First commit for a task must reference the spec file:**
  - `[BK-TASK-042][io] Implement spec per docs/specs/BK-TASK-042.md`
- Examples:
  - `[BK-TASK-042][io] Fix spread sign negation in clean_tables.py`
  - `[BK-TASK-043][features] Add home_team column preservation in rolling_features`

### Tag Rules

- Pre-change checkpoint (always tied to ticket): `pre-BK-TASK-###-YYYYMMDD`
- Post-verified milestone: `v2.X-BK-TASK-###-verified`
- Tags are never deleted
- Verified milestone tags can never be overwritten or moved

### Test Requirements

- `pytest` must pass before any merge to main
- New code requires new tests
- Test failures block all forward progress until resolved

---

## Section 4: Session Hygiene

### Session Reset Rule

Before each session with any LLM agent, state explicitly:

> "This session begins with no prior context. You have no memory of previous conversations."

This prevents the agent from hallucinating continuity.

### Scope Limits

- One task per session
- Task must be completable in single session
- No "while we're here, let's also..."
- If scope expands, start new session with new spec

### Context Rules

- Each session starts with explicit context (not accumulated)
- Provide only: task spec, relevant file contents, recent terminal output
- Do not rely on "Claude remembers" — assume fresh start
- Project knowledge base contains only verified reference docs

### Session End Protocol

1. State what was accomplished
2. State what was verified (with evidence)
3. State what remains to be done
4. Update documentation before closing

---

## Section 5: Document Management

### Source of Truth Hierarchy

1. **Code** — always the ultimate truth
2. **Test results** — verified behavior
3. **Terminal output** — observed state
4. **Documentation** — describes code, not intentions

### Document Rules

- Docs describe actual state, never intended state
- If doc and code conflict, doc is wrong
- After any code change, doc update is required before task is "done"
- Stale docs are worse than no docs — delete rather than leave outdated

### Required Documents

| Document | Purpose | Update Trigger |
|----------|---------|----------------|
| ARCHITECTURE.md | System diagram, module dependencies, data flow | Any structural change |
| SCHEMA_GAME.md | Locked schema for test_games | Any schema change |
| SCHEMA_UPSTREAM.md | Clean table schemas | Any ingestion change |
| DATA_SOURCES.md | Data file locations and formats | Any data path change |
| TEST_MATRIX.md | All tests, what they verify, what they guard | Any test addition/removal |
| PROJECT_LOG.md | Running log of completed tasks with ticket IDs | Every task completion |
| KNOWN_ISSUES.md | Current bugs and limitations | Any new issue discovered |

### PROJECT_LOG.md Entry Format

Each entry must include:

```
## BK-TASK-###: [title]
**Date:** YYYY-MM-DD
**Spec:** docs/specs/BK-TASK-###.md
**LLM Logs:** data/llm_logs/BK-TASK-###_planner_*.json, data/llm_logs/BK-TASK-###_auditor_*.json
**Tags:** pre-BK-TASK-###-YYYYMMDD, v2.X-BK-TASK-###-verified (if applicable)
**Summary:** [what was done]
```

---

## Section 6: Failure Protocol

When something doesn't work:

### Step 1: STOP

- Do not attempt fixes
- Do not propose solutions
- Do not continue with other work

### Step 2: Gather Evidence

- Exact error message
- Full stack trace
- Command that was run
- State of inputs (file contents, data shapes)

### Step 3: Diagnose

- Identify the specific line/function that failed
- Identify what input caused the failure
- Identify what the input should have been
- Trace back to find where the wrong state originated
- **During diagnosis, Planning Agent and Auditor must not propose fixes. They may only describe what happened and why.**

### Step 4: Confirm Diagnosis

- Planning Agent states diagnosis with citations
- Project Lead runs verification command to confirm
- If diagnosis is wrong, return to Step 2

### Step 5: Specify Fix

- Only after diagnosis is confirmed
- Fix targets root cause, not symptoms
- Spec includes verification test

### Step 6: Resume Normal Flow

- Return to Section 2, Phase 1 with fix as the new task

### Step 7: Postmortem (Required)

After any failure is fixed, add a postmortem entry to PROJECT_LOG.md:

```
## Postmortem: BK-TASK-###

**What broke:** [specific failure]
**Why it broke:** [root cause]
**What systemic weakness allowed it:** [process gap]
**Prevention measure added:** [new check or rule]
```

---

## Section 7: Citation Requirements

Every claim must be backed by evidence. No exceptions.

### Burden of Proof Rule

**Any claim made without citation is automatically treated as false until proven with evidence.**

This shifts the burden of proof off the Project Lead. If an agent says "this works" without showing test output, assume it doesn't work.

### UNVERIFIED Marker

If an agent cannot produce evidence for a claim, it must mark it as:

```
UNVERIFIED: [claim]
```

Any statement marked UNVERIFIED is non-actionable. You can search logs for this string to find gaps.

### Claim Types and Required Citations

| Claim Type | Required Citation |
|------------|-------------------|
| "This code does X" | File path, line number, quoted code |
| "The data contains X" | Terminal output showing data |
| "The test passes/fails" | pytest output |
| "The schema requires X" | Schema file, line number |
| "This value is X" | Command and output that produced it |
| "Research shows X" | Document name, section, quote |

### Forbidden Phrases

These phrases are not allowed without immediate citation:

- "should be"
- "typically"
- "I believe"
- "probably"
- "usually"
- "in most cases"
- "this will work"
- "this is correct"

### Citation Format

```
Claim: [statement]
Evidence: [file:line] or [command -> output]
```

Example:
```
Claim: The spread sign is negated during cleaning.
Evidence: ball_knower/io/clean_tables.py:322
          df["market_closing_spread"] = -pd.to_numeric(df["closing_line"], errors="coerce")
```

---

## Section 8: Outside Auditor Protocol

### Auditor Identity

- The Outside Auditor must be GPT (different model family than Planning Agent)
- This provides actual independence, not just fresh context

### When to Use Auditor

- Before any merge to main
- After any multi-file change
- When Planning Agent expresses uncertainty
- When Project Lead requests second opinion
- After any failure/fix cycle

### Mandatory Re-Audit Condition

**Any revision of a previous conclusion requires re-audit.**

If the Planning Agent changes its assessment of something it previously approved, the Auditor must review before proceeding. This prevents LLMs from "changing their mind" silently.

### Auditor Briefing Format

Provide to Auditor:
1. Task description (1-2 sentences)
2. Files changed (list)
3. Test results (pytest output)
4. Any specific concerns

Do NOT provide:
- Planning Agent's analysis
- Project history
- Conversation context
- Assumptions or interpretations

### Auditor Review Process

1. Auditor requests any needed file contents
2. Auditor requests any needed diagnostic commands
3. Project Lead runs commands, provides output
4. Auditor states findings with citations
5. Auditor gives: APPROVED, APPROVED WITH NOTES, or BLOCKED

### Auditor Authority

- BLOCKED means work does not proceed
- Auditor can request any evidence
- Auditor's skepticism is a feature, not a bug
- Disagreement between Planning Agent and Auditor triggers Project Lead decision

---

## Section 9: Recovery Protocol

### Before Any Risky Operation

"Risky" means: multi-file changes, refactors, data regeneration, branch operations

1. Create checkpoint tag: `git tag checkpoint-YYYYMMDD-HHMM`
2. Verify tag: `git log --oneline -1`
3. Record in session notes

### When Things Go Wrong

1. Assess damage scope
2. If recoverable with fix: follow Failure Protocol
3. If unclear or extensive: revert to checkpoint

### Revert Procedure

```bash
git log --oneline [checkpoint-tag]
git checkout [checkpoint-tag] -- .
# Or hard reset (loses all changes since checkpoint)
git reset --hard [checkpoint-tag]
```

### After Revert

1. Confirm state: `git status`, `pytest`
2. Document what went wrong
3. **No further development may happen until a fresh BK-TASK is opened documenting:**
   - Why the revert was necessary
   - What work will be redone under the new ticket
4. Start new session with lessons learned

---

## Section 10: Quick Reference

### Starting a Task

```
1. Assign ticket ID: BK-TASK-###
2. Write spec with preconditions and success criteria
3. Verify preconditions with terminal output
4. Tag: git tag pre-BK-TASK-###-YYYYMMDD
5. Create branch: git checkout -b feature/BK-TASK-###-short-name
6. Implement via API call to planner, then to implementation agent
7. Test: pytest --tb=short
8. Get auditor sign-off via API call to GPT
9. Merge (Project Lead only), delete branch, update docs
10. Tag if milestone: v2.X-BK-TASK-###-verified
```

### When Stuck

```
1. STOP
2. Gather evidence (error, trace, inputs)
3. Diagnose with citations via API call
4. Confirm diagnosis with verification command
5. Spec the fix
6. Resume normal flow
7. Write postmortem after fix is complete
```

### Forbidden Actions

```
- Skipping precondition verification
- Implementing without spec approval
- Merging without tests passing
- Merging without auditor sign-off
- Merging by anyone other than Project Lead
- Deleting branches with unmerged work
- Making claims without citations
- Proceeding after failure without diagnosis
- Using chat interfaces (Claude.ai, ChatGPT web) for binding planning or audit decisions
- Overwriting verified milestone tags
- Referencing previous chat context
- Proposing fixes during diagnosis phase
```

### API Commands Cheat Sheet

```bash
# Planning (Claude)
python scripts/llm_api.py --role planner --task "BK-TASK-###" --context context.txt --prompt "..."

# Auditing (GPT)
python scripts/llm_api.py --role auditor --task "BK-TASK-###" --context context.txt --prompt "..."

# View logs
ls -la data/llm_logs/
cat data/llm_logs/BK-TASK-###_*.json
```

---

## Section 11: API Execution Layer

This section provides the technical setup for making LLM API calls without chat drift.

### Why This Matters

Chat interfaces (Claude.ai, ChatGPT web) accumulate context invisibly. The LLM "remembers" things you didn't explicitly provide, leading to drift and hallucinations. The API execution layer ensures:

- Only whitelisted context is sent
- Every input/output is logged
- Operations are reproducible
- No hidden state accumulates

### Setup (One-Time)

#### Step 1: Get API Keys

**For Claude (Planning Agent):**
1. Go to https://console.anthropic.com/
2. Create an account or sign in
3. Go to API Keys, then Create Key
4. Save the key securely (you will only see it once)

**For GPT (Auditor):**
1. Go to https://platform.openai.com/
2. Create an account or sign in
3. Go to API Keys, then Create new secret key
4. Save the key securely

#### Step 2: Set Up Environment Variables in Codespaces

In your Codespaces terminal:

```bash
echo "ANTHROPIC_API_KEY=your-claude-key-here" >> .env
echo "OPENAI_API_KEY=your-gpt-key-here" >> .env
```

#### Step 3: Install Required Packages

```bash
pip install anthropic openai python-dotenv
```

#### Step 4: Create the Execution Script

The script is located at `scripts/llm_api.py`. See that file for implementation.

### Usage

#### For Planning (Claude):

1. Create a context file with relevant information
2. Run the planner:
```bash
python scripts/llm_api.py --role planner --task "BK-TASK-001" --context context.txt --prompt "Your instructions here"
```

#### For Auditing (GPT):

1. Create context with the proposed change and evidence
2. Run the auditor:
```bash
python scripts/llm_api.py --role auditor --task "BK-TASK-001" --context context.txt --prompt "Review this fix."
```

### Log Structure

All API calls are logged to `data/llm_logs/` with full input/output. This provides:

- Complete audit trail
- Reproducibility (same input = same prompt)
- Evidence for postmortems
- Protection against "I never said that" disputes

**Every BK-TASK must have at least one planner log and one auditor log in `data/llm_logs/`, or a note in PROJECT_LOG.md explaining why an LLM was not used.**

### Configuration

Model names and system prompts are currently hardcoded in the script. For future flexibility, consider extracting these to `configs/llm_config.json`. For now, update the model string directly in the script when newer models are available.

### When to Use API vs Chat

| Use API | Use Chat |
|---------|----------|
| Any planning task | Initial exploratory research |
| Any audit task | Learning how something works |
| Any specification | Brainstorming (non-binding) |
| Any implementation instruction | Asking for explanations |
| Anything that will be acted upon | Nothing that will be acted upon |

**Rule of thumb:** If the output will result in code changes, file changes, or decisions — use the API.

### Fallback for Urgent Issues

If the API setup is not ready and you need immediate help:

1. Start a fresh chat (not in any Project)
2. Begin with: "This session begins with no prior context. You have no memory of previous conversations."
3. Provide all context explicitly in the first message
4. Screenshot or copy the entire exchange for logs
5. Treat all outputs as unverified until confirmed

This is a temporary measure, not standard practice.

---

*This protocol is the operating agreement for Ball Knower v2 development. Deviations require explicit Project Lead approval with documented reasoning.*
