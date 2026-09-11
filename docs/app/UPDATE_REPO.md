# How to Update the Repository

The latest version of the code now lives on the `main` branch. To pull the latest changes:

## Quick Update (3 steps)

Open a terminal in the project folder and run:

```bash
git checkout main
git pull origin main
```

That's it. You should see something like:

```
Updating abc1234..def5678
Fast-forward
 app.py             |  10 +-
 itanet_integration.py |  50 +++-----
 ...
```

If it says **"Already up to date."**, you already have the latest version.

---

## Step-by-step (if the quick version didn't work)

### 1. Check which branch you're on

```bash
git branch
```

The branch with `*` next to it is your current branch.

### 2. Switch to `main`

If you're not on `main`, switch to it:

```bash
git checkout main
```

You should see: `Switched to branch 'main'`.

### 3. Pull the latest changes

```bash
git pull origin main
```

### 4. Confirm everything's clean

```bash
git status
```

You should see: `nothing to commit, working tree clean`.

---

## If something goes wrong

Send me the output of these three commands and I'll take a look:

```bash
git status
git log --oneline -3
git remote -v
```
