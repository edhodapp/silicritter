# silicritter — project-local Claude instructions

Analog-neural-silicon exploration, licensed AGPL-3.0-or-later. Aim: silicon that emulates living-creature neural networks, with developmental dynamics (synaptic exuberance + lifelong sculpting), not just adult steady-state plasticity.

## Start here
- `DECISIONS.md` — architectural decision log (immutable entries, D001+).
- `SESSION-HANDOFF.md` — origin context from the 2026-04-21 session that spawned this project.
- `~/.claude/projects/-home-ed-silicritter/memory/` — durable project memory (framing, design sketches in progress).

## Mode
Experimental / blue-sky per global `~/.claude/CLAUDE.md`. **Not a deliverable.** No CD pipeline, build scaffolding, or directory skeleton predicted in advance — structure emerges per experiment. Review discipline still applies to functional commits (global pre-commit pipeline: quality gates + Gemini review + clean-Claude review).

Blue-sky collaboration mode is active: Ed brings intuition, framing, taste; I bring literature review, prior art, formal apparatus. Don't push Ed to "do his own research first" and don't police focus when adjacent-project threads interleave.

**Every branch implies an assertable behavior; every test must assert (HARD RULE).**

- Branch coverage is a floor, not a contract. A branch can be
  executed without anything checking that it produced the right
  result; the coverage report ticks the line as covered while no
  behavioral contract is pinned.
- A test without an `assert` (or `pytest.raises`, `np.testing.assert_*`,
  equivalent) is not a test. It's an execution-only code path that
  catches nothing. Fixtures don't need assertions; test functions do.
- The deeper principle: every branch in the code exists because it
  produces a different observable outcome for different inputs. If
  you can't identify what to assert about a branch, the branch
  shouldn't exist — either it's dead code, or the "different outcome"
  hasn't been articulated. The act of writing the assertion forces
  you to name the behavior.
- Therefore: 100% branch coverage with shallow tests is worse than
  80% branch coverage with deep tests. Coverage gates are necessary
  but not sufficient; functional contract pinning is what catches
  regressions.

When writing or reviewing a test, ask: "for each branch this code
path traverses, what observable behavior does the test pin? If I
mentally inverted the branch's logic, would this test fail?" If the
answer is no, the test isn't pinning the contract.

**The "no return-value difference" trap.** Sometimes a branch's
only observable difference is a side effect — a printed error, a
log message, a write to stderr. That side effect IS the assertable
behavior; the test doesn't get to skip the assertion just because
the function's return value is unchanged. pytest provides
fixtures for capturing stdio:
- `capsys` — captures `sys.stdout` / `sys.stderr` as text.
- `capfd` — captures file-descriptor-level output (1, 2), useful
  for code that writes to fd directly or via subprocess.
- `caplog` — captures `logging` records.

Use them; assert on the captured output. "The branch only prints"
is not a license to skip the test, it's a directive to use the
right fixture.

**Error/exception paths must do what you expect.** Every error path
needs a test that exercises it AND asserts on its observable output
— not just "an exception was raised" but the *content* of the
message, log line, or returned error value. When error paths land
without being tested, two failure modes follow:

1. **The error path itself raises an unhandled exception** (typo in
   format string, calling a method on a None value, dereferencing a
   field that doesn't exist on the error case's data). The user's
   actual error gets replaced by an opaque crash from inside the
   error handler.
2. **The output doesn't help debug the issue.** A message like
   "validation failed" without the bad value, the expected format,
   or where the validation came from leaves the user reading code
   to figure out what happened.

So an error-path test asserts at least:
- The error type is what you said it is (`pytest.raises(SpecificError)`).
- The error message includes the *input that was wrong* so the user
  can locate it in their config / call.
- The error message includes *what was expected* so the user knows
  how to fix it.
- If the error path produces logs, stdio, or other side effects,
  those are captured (via `capsys` / `caplog`) and asserted on too.

**`python -O` / `-OO` are unsupported.** The codebase relies on
`assert` statements to enforce runtime invariants (e.g.
`assert n_steps % WINDOW_STEPS == 0`), and several module docstrings
are read at import time (e.g. `argparse(description=__doc__.splitlines()[0])`).
Running under `-O` silently disables invariant enforcement; `-OO`
additionally crashes the docstring lookups. Both modes break
correctness contracts the codebase depends on. Defensive `__doc__ or
""` patches at individual call sites would imply support for `-OO`
that the codebase as a whole doesn't honor — the rigorous response is
to declare the modes unsupported (here) and let any future
deliverable-track refactor convert asserts to explicit raises if it
needs `-O` compatibility.

## Review findings: every one gets walked through with Ed (HARD RULE)

Every finding from the Gemini hook or the clean-Claude reviewer must be
reviewed with Ed before the commit lands. There are no silent
deferrals.

- "Pre-existing" is not a license to skip. A finding being older than
  the current diff does not make it less of a defect; it just means it
  has been deferred longer.
- "Out of scope" is Ed's call, not Claude's. Claude does not
  unilaterally classify a finding as out-of-scope to ship a commit
  faster.
- The valid response to a review with N findings is to list ALL N,
  classify each transparently as (a) will fix in this commit, (b) want
  Ed's input on intent before changing, or (c) Ed-confirmed defer with
  a reason — and **wait for Ed's sign-off on (b) and (c) before
  committing**.
- This applies retroactively too: if a review surfaces an issue the
  commit didn't introduce but is in the same class as the change being
  made (e.g. dropping unused rasters in `_training_scan` while leaving
  the same pattern intact in `_measure_fitness`), addressing it in the
  same commit is the default, not the exception.

Why the rule exists: skipping findings without explicit permission
silently lowers the project's quality bar. The Gemini + clean-Claude
review pair is the project's quality gate, and the cost of pausing to
walk through findings is much smaller than the cost of accumulated
deferred-debt that nobody comes back to.

## License and posture
silicritter is licensed **GNU AGPL-3.0-or-later** (D004, supersedes D002). Ed Hodapp is sole author and sole copyright holder. **No external contributions accepted** — pull requests and patches are declined to preserve consolidated copyright ownership (so Ed retains the right to relicense, dual-license for commercial clients, or open to contributions under a CLA later). The AGPL-as-consulting-driver pattern is explicit: commercial users needing to deploy without AGPL obligations route to Ed for a commercial-license / consulting engagement.

Repository is **public** on GitHub at `edhodapp/silicritter`, flipped from private to public on 2026-04-22 (D005). With AGPL-3.0-or-later in place and the no-external-contributions policy explicit, there was no load-bearing reason to keep it private.

- Any MPW shuttle submission (TinyTapeout, Efabless chipIgnite, etc.) still requires Ed's explicit go-ahead — license compatibility is a solved problem, but Ed owns the when/where call.
- Commercial clients who cannot accept AGPL's network-copyleft route to Ed for a commercial license / consulting engagement.

## Downstream directionality
AGPL code cannot be pulled into proprietary sibling projects under BSD-style terms without violating the AGPL. silicritter can serve as reference / teaching material for Ed's proprietary projects, but code does not flow from silicritter into proprietary repos without Ed explicitly relicensing (which he can do unilaterally as sole copyright holder, if and when he chooses).
