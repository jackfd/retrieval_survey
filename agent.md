# Implementation Agent Policy

## 1. Role and Scope

This document governs the **implementation agent only** for the index input pipeline.

Primary objective:

- implement `index_scheduler.py`, `build_index_inputs.py`, and supporting config/artifacts
- strictly comply with [contract.md](./contract.md) and [docs/technical_design.md](./docs/technical_design.md)

## 2. Allowed Scope

The implementation agent MAY:

1. Add or modify runtime code required by approved design.
2. Add tests and fixtures needed to verify contract compliance.
3. Update non-breaking documentation clarifications.
4. Improve observability where it does not alter contract behavior.

## 3. Prohibited Scope

The implementation agent MUST NOT:

1. Change public CLI interfaces defined in contract without approved contract revision.
2. Change output schema, file names, or output directory contract without approval.
3. Introduce non-contract retry semantics.
4. Add hidden fallback logic that violates strict 768-dim rule.
5. Start implementation before Design Gate and Contract Gate are `PASS`.

## 4. Mandatory Delivery Checklist

Every implementation PR/change set MUST include explicit evidence for:

1. **Contract compliance**
   - CLI args exactly match contract
   - output schema exactly match contract
2. **Schema checks**
   - docs parquet fields and vector dimension checks
   - queries parquet fields and vector dimension checks
3. **Logging checks**
   - `app.log` path correctness
   - exception class and message presence
4. **Retry checks**
   - failed-doc-only reprocessing
   - merge-by-`doc_id` overwrite behavior

If any checklist item is missing, delivery is incomplete.

## 5. Change Protocol

When a contract-breaking need is discovered:

1. Stop implementation for impacted scope.
2. Propose contract update in `contract.md` with concrete rationale.
3. Update impacted design section in `docs/technical_design.md`.
4. Wait for explicit approval.
5. Resume implementation only after approval.

## 6. Definition of Done

A task is done only when all are true:

1. Design Gate status: `PASS`
2. Contract Gate status: `PASS`
3. Implementation Gate status: `PASS`
4. Validation Gate status: `PASS`
5. All acceptance criteria in contract Section 8 are satisfied

Any unresolved contract deviation means `NOT DONE`.

## 7. Review Artifacts Required from Agent

Implementation agent must provide:

1. Changed file list
2. Contract traceability notes (requirement -> code/test evidence)
3. Test execution summary (pass/fail and coverage of critical scenarios)
4. Known risks or deferred items with explicit rationale
