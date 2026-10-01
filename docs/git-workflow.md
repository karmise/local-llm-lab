# Git Workflow

The Git repository root is `local-llm-lab`. The `automation` directory is a
Python project inside that repository. Use the `main` branch for the initial
commit; subsequent changes can use feature branches.

## First commit

From `local-llm-lab`, after the GitHub remote has been connected:

```bash
git status
git remote -v
git add .
git diff --cached --stat
git diff --cached
git commit -m "Initialize LLM RAG testing project"
git push -u origin main
```

Review the staged changes before committing. Runtime secrets, virtual
environments, IDE settings, reports and local application exports are excluded
by `.gitignore`. Configuration examples and fictional test data are included.

If Git reports a missing author identity, configure it for this repository:

```bash
git config user.name "Your Name"
git config user.email "YOUR_GITHUB_COMMIT_EMAIL"
```

Replace the placeholders with your chosen commit identity. A GitHub-provided
no-reply address can be used as the commit email.

## Subsequent changes

```bash
git switch -c feature/workspace-api
# Make and verify the change.
git add automation docs
git diff --cached
git commit -m "Add workspace API operations"
git push -u origin feature/workspace-api
```

Open a pull request on GitHub to review and merge the branch. In PyCharm,
the Commit and Push actions use the same repository and GitHub remote.
