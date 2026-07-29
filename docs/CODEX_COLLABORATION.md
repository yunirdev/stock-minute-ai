# Codex collaboration contract

This document keeps work small, verifiable, and additive. It prevents repeated
repository-wide reviews caused by mixing a trading change, research change, UI
change, and legacy cleanup into one request.

## Default task unit

One request should change or assess one vertical slice. A slice has a single
user-visible result, a bounded set of production callers, and a clear test
boundary.

Use this request format:

```text
Task type: review | diagnose | implement | cleanup
Slice: <one flow, e.g. Runtime -> RiskEngine -> OrderIntentStore>
Goal: <one observable result>
Allowed files: <files/directories, or "discover within slice">
Do not: <e.g. no UI work, no legacy cleanup, preserve uncommitted changes>
Acceptance: <tests/checks and expected outcome>
Context: <prior decision or issue number, if any>
```

If any field is missing, Codex should make the smallest safe assumption and
state it before changing code. It must ask before expanding the slice into a
different subsystem.

## Work modes

### Review

- Read-only unless the request explicitly asks for a report file.
- Report only findings in the requested slice, grouped as P0/P1/P2.
- Each finding needs an affected production path, evidence, impact, and a
  smallest viable fix.
- Do not rerun a full architecture review merely because adjacent code exists.

### Diagnose

- Reproduce or trace one observed failure.
- Do not fix it unless implementation is explicitly included.
- Finish with cause, confidence, affected scope, and a proposed next task.

### Implement

- Preserve unrelated dirty-worktree changes.
- Modify only files required by the agreed slice.
- Add or adjust regression tests for changed behavior.
- Run focused tests first, then the agreed verification level.

### Cleanup

- First classify targets as production, research/UI, legacy/manual, or generated.
- Move or delete only targets proved unused by production callers and tests.
- Cleanup never includes behavior changes unless separately requested.

## Production ownership map

| Area | Canonical path | Boundary |
|---|---|---|
| Daily research | `daily_research.py` -> `ai_states.duckdb` | Analysis only; no broker import |
| Trading | `main.py` -> `runtime.py` -> decision/risk/order store -> `broker/alpaca.py` | Runtime is the sole order submitter |
| Monitoring | `monitor_nice.py` / `monitor_data.py` / status sidecar | Must not make trade decisions |
| Research/manual tools | `ai/`, `teams/`, `strategies/`, `backtest/`, notebooks | Not a Runtime dependency unless explicitly promoted |

When a change crosses two rows, split it into two tasks unless the interface
between them is the stated goal.

## Required handoff

Every completed implementation response contains:

1. Outcome and changed files.
2. Verification actually run and results.
3. Remaining risks or intentionally untouched work.
4. One recommended next smallest task.

For a review, replace changed files with the reviewed scope and findings.

## Decision records

Create or update `docs/decisions/` only for material choices that affect the
production boundary, data ownership, safety, or task workflow. A record must
state context, decision, consequences, and superseded alternatives. Do not use
`AGENTS.md` as a chronological work log.

## Explicit triggers for a full review

A repository-wide architecture review is appropriate only when at least one of
these is true:

- a production entrypoint or broker changes;
- persistent data ownership/schema moves between components;
- a safety invariant changes;
- the user asks for a release-readiness or security review.

Otherwise review the smallest applicable vertical slice.
