# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `evaluate.py` — fixed constants, environment factory, and evaluation harness. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs for a **fixed time budget of 12 minutes** (wall-clock training time). Launch it as:

```
python train.py > run.log 2>&1
```

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, network size, etc.

**What you CANNOT do:**
- Modify `evaluate.py`. It is read-only. It contains the fixed evaluation harness, environment factory, and training constants (`TIME_BUDGET`, `ENV_ID`, `NUM_EVAL_EPISODES`).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml` (torch, gymnasium[box2d], numpy, pandas, matplotlib).
- Modify the evaluation harness. The `evaluate_policy` function in `evaluate.py` is the ground truth metric.

**The goal is simple: get the highest `avg_return`.** BipedalWalker-v3 is considered "solved" at an average return of ~300 over 100 episodes.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline — run the training script as-is.

## Output format

Once the script finishes it prints a summary like this:

```
---
avg_return:       135.562469
std_return:       95.432100
min_return:       -112.340000
max_return:       312.450000
total_timesteps:  4194304
training_seconds: 720.1
total_seconds:    728.3
num_updates:      128
```

Extract the key metric:

```
grep "^avg_return:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 4 columns:

```
commit	avg_return	status	description
```

1. git commit hash (short, 7 chars)
2. avg_return achieved (e.g. 135.562469) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	avg_return	status	description
a1b2c3d	135.562469	keep	baseline PPO
b2c3d4e	180.450000	keep	increase NUM_ENVS to 16
c3d4e5f	120.000000	discard	remove advantage normalization
d4e5f6g	0.000000	crash	attempted SAC (missing imports)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^avg_return:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If avg_return improved (higher), you "advance" the branch, keeping the git commit
9. If avg_return is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take ~12 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 20 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes, use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes ~12 minutes then you can run approx 5/hour, for a total of about 40 over the duration of the average human sleep. The user then wakes up to experimental results!
