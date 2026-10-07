# Contributing

## Setup

vLLM runs on Linux only. On Windows, work inside WSL2. You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```
uv sync                  # dev tools, without vLLM
uv sync --extra vllm     # adds vLLM, needed for GPU tests
```

## Checks

```
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest            # CPU tests only
uv run pytest -m gpu     # tests that start vLLM on a CUDA GPU
```

CI runs all of these except the GPU tests on every push to main and every pull request.

## Code

Add a module when the existing ones can no longer hold the code cleanly, and an abstraction when it has a second user.

A change to how something is measured goes in its own commit. Results measured the old way are not compared with results measured the new way.

Comments state constraints and reasons the code cannot show. Don't narrate what the code does, and don't record history in comments ("changed from X"); git log holds history. Public functions get a docstring.

Tests run on CPU by default. Mark anything that starts vLLM with `@pytest.mark.gpu`.

## Writing

Every claim about speed or quality in the README or docs gives the number, the GPU and the model, or links to the measurement. Commands shown in the README must exist and their output must be real. Use sentence-case headings, no emoji and no rows of badges.

## Commits and pull requests

Commit subjects follow `area: what changed`, in the imperative, lowercase after the colon, with no trailing period, for example `harness: retry engine start after OOM`. Add a body only when the reason is not obvious from the diff.

A pull request description covers the problem, why this approach, what it does not handle, and benchmark numbers when performance changes.
