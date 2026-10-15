# Git workflow

The Git repository root is `local-llm-lab`; `automation` is a Python project
inside it. `main` is protected, so every change goes through a pull request.

## Make a change

```bash
git switch main && git pull
git switch -c refactor/short-description
# Make the change, then run the offline checks from automation:
cd automation
python -m pytest tests/unit -q
python -m ruff check .
python -m isort --check-only --diff .
python -m yapf --diff --recursive .
cd ..
git add -A automation docs
git diff --cached
git commit -m "refactor: describe the change"
git push -u origin refactor/short-description
gh pr create --base main
```

Commit messages use conventional prefixes: `feat:`, `fix:`, `docs:`, `refactor:`,
`test:`, `ci:` and `style:`. Review staged changes before committing.
`.gitignore` excludes runtime secrets, virtual environments, IDE settings,
reports and local application exports. Configuration examples and fictional test
data are tracked. The repository is public, so never commit keys or personal notes.

## Merge a pull request

`main` requires three checks:

| Check | Source | When it runs |
| --- | --- | --- |
| `Ruff and unit tests` | `framework-checks.yml` | Automatically on every push |
| `Offline Page Object checks` | `framework-checks.yml` | Automatically on every push |
| `RAG benchmark quality` | `rag-live-quality.yml` ("Fresh RAG quality") | Manually, for the pull request |

Start the live check from `main` with the pull request number. It tests the
pull request's merge commit and takes about two hours on a CPU runner:

```bash
gh workflow run rag-live-quality.yml --ref main -f pull-request=<number> -f profile=curated -f models=primary
```

A `smoke` run publishes a different status and cannot satisfy the merge gate.
Every new commit needs a new run, and a failed run is investigated, not repeated.
When all three checks pass, squash-merge the pull request.

The live workflow runs main's version of the workflow files. A pull request
that changes those files, for example test paths they call, cannot be checked by
its own gate. In that case an administrator merge is a deliberate maintenance
exception, followed by a gate run on `main` to confirm the change. It is never a
quality pass.
