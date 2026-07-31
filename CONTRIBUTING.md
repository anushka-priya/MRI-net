# Contributing to MRI-net

Thanks for jumping in. Here's how this works.

## Before you start

1. Check the [Issues tab](../../issues) and pick one that's still open and
   unassigned.
2. Comment on the issue saying you're working on it as this avoids two people
   independently fixing the same bug.
3. If you're not sure your understanding of the bug is right, ask in the
   issue thread before writing code. Better to check early than debug the
   wrong thing for an hour.

## Setup

```bash
git clone https://github.com/anushka-priya/MRI-net.git
cd MRI-net
uv sync
uv run python -m MRI_net.main train
```

See the main [README](README.md) on how to install `uv` if you don't have it yet.

## Making your fix

1. Fork the repo.
2. Create a branch off `main`, named descriptively:
```bash
   git checkout -b fix/classification-report-mismatch
```
3. Make your fix. Keep it scoped to the one issue you picked. Don't bundle
   unrelated changes, refactors, or style fixes into the same PR.
4. Test locally. At minimum, confirm `uv run python -m MRI_net.main train`
   runs end to end without errors, and that the specific symptom described in
   the issue is actually gone (not just "it looks different now").
5. Commit with a clear message describing what was wrong and what changed.

## Opening the PR

- Open your PR against `main`.
- Reference the issue number in the description (e.g. `Fixes #3`).
- Briefly explain: what was actually broken, why, and how your change fixes
  it.
- Include before/after evidence wherever relevant — a metric, a plot, a couple
  of print statements. Show that it's actually fixed, not just changed.

## Review

- A maintainer will review and may ask for small changes which is normal,
  not a rejection.
- Please be patient with turnaround time; this is run alongside other work.
- Once approved, we'll merge it.

## Ground rules

- One bug per PR.
- Don't paste bug locations or fixes into other people's open issue threads.
- If you spot a *new* bug that isn't already tracked, feel free to open a
  fresh issue for it.
