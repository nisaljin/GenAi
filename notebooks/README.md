# Notebooks README

This folder contains documentation and experimentation notebooks for the Foley pipeline.

## 1. Notebook Index

## 1.1 `agentic_foley_system_deep_dive.ipynb`

Purpose:
- Teammate onboarding and architecture understanding.

Covers:
- system components and interactions
- websocket and backend flow
- agentic loop behavior and decisions
- event taxonomy
- failure modes and triage checklist
- runbook commands

Use this first if you are new to the project.

## 1.2 `foley_pipeline_stage_test.ipynb`

Purpose:
- stage-level testing and experimentation.

Typical use:
- isolate one stage behavior
- verify assumptions quickly
- prototype prompt/model changes before moving into production code

## 2. Suggested Reading Order

1. `agentic_foley_system_deep_dive.ipynb`
2. `foley_pipeline_stage_test.ipynb`

## 3. Collaboration Rules

- Architecture/process explanations belong in the deep-dive notebook.
- Experimental/testing content belongs in stage-test notebook.
- If event schema or flow changes, update deep-dive sections on:
  - flow diagrams
  - event taxonomy
  - troubleshooting

## 4. Maintenance Checklist

When core logic changes (e.g., new event types, retry policy, perception behavior):

1. Update notebook narrative to reflect new behavior.
2. Add a short “what changed” section/date in the relevant notebook.
3. Ensure command examples still match current scripts.
4. Confirm paths and filenames exist in repo.

## 5. Related Documentation

- Root overview: `../README.md`
- Backend internals: `../server/README.md`
- Frontend runtime: `../web/README.md`
- Stage test scripts: `../scripts/README_STAGE_TESTS.md`
