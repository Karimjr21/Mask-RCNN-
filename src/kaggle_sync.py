"""Optional: mirror training checkpoints to a Kaggle Dataset during a run.

This is a **no-op unless** the environment variable ``KAGGLE_CKPT_DATASET`` is
set, so local training is completely unaffected. On Kaggle, set it to e.g.
``karimahmed21/isaid-checkpoints`` and the trainer pushes ``last.pth`` and
``model_best.pth`` as a new dataset version every ``KAGGLE_CKPT_INTERVAL``
epochs (default 2).

Why: a Kaggle GPU session can disconnect or hit the 12-hour limit mid-run, and
``/kaggle/working`` is wiped when it ends. Mirroring the checkpoints to a
dataset means a disconnect costs at most a few epochs — re-attach the dataset
next session, restore ``last.pth``, and ``train.py`` resumes automatically.

Env vars:
    KAGGLE_CKPT_DATASET   "owner/slug" of the checkpoints dataset (enables sync)
    KAGGLE_CKPT_INTERVAL  push every N epochs (default "2")
"""
import os
import sys
import json
import shutil
import subprocess


def _interval():
    try:
        return max(1, int(os.environ.get("KAGGLE_CKPT_INTERVAL", "2")))
    except ValueError:
        return 2


def maybe_sync_checkpoints(epoch, checkpoint_dir, log_file=None, force=False):
    """Push key checkpoints to the Kaggle dataset every N epochs.

    Curated to ``last.pth`` (full resume state) + ``model_best.pth`` (best
    model) so each version stays ~0.5 GB rather than dragging the redundant
    per-epoch checkpoints along. Any failure is caught and reported — checkpoint
    mirroring must never crash a training run.
    """
    dataset = os.environ.get("KAGGLE_CKPT_DATASET")
    if not dataset:                                  # disabled (e.g. local run)
        return
    if not force and epoch % _interval() != 0:
        return

    parent = os.path.dirname(os.path.normpath(checkpoint_dir)) or "."
    stage = os.path.join(parent, "_kaggle_sync")
    os.makedirs(stage, exist_ok=True)

    staged_any = False
    for name in ("last.pth", "model_best.pth"):
        src = os.path.join(checkpoint_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(stage, name))
            staged_any = True
    if log_file and os.path.isfile(log_file):
        shutil.copy2(log_file, os.path.join(stage, os.path.basename(log_file)))
    if not staged_any:                               # nothing to push yet
        return

    # dataset-metadata.json tells `kaggle datasets version` which dataset to bump
    with open(os.path.join(stage, "dataset-metadata.json"), "w") as f:
        json.dump({"id": dataset, "title": dataset.split("/")[-1]}, f)

    cmd = [sys.executable, "-m", "kaggle", "datasets", "version",
           "-p", stage, "-m", f"epoch {epoch}", "--quiet"]
    try:
        subprocess.run(cmd, check=True)
        print(f"  ⤴ Mirrored checkpoints to Kaggle dataset "
              f"{dataset} (epoch {epoch})")
    except Exception as exc:                         # never break training
        print(f"  ⚠ Kaggle checkpoint sync failed (epoch {epoch}): {exc}")
