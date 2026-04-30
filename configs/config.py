import os
import subprocess


def _repo_root() -> str:
    """Return the main repo root — works from both the main repo and git worktrees."""
    cwd = os.path.dirname(os.path.abspath(__file__))
    try:
        git_common = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        # git may return a relative path; resolve it against the cwd we used
        git_common_abs = os.path.normpath(os.path.join(cwd, git_common))
        return os.path.dirname(git_common_abs)
    except Exception:
        return os.path.dirname(cwd)


_ROOT = _repo_root()

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_ROOT  = os.path.join(_ROOT, "data", "BUSI")
SAVE_PATH  = os.path.join(_ROOT, "checkpoints", "best_model.pth")
LOG_PATH   = os.path.join(_ROOT, "logs", "training_log.csv")

# ─── Data ─────────────────────────────────────────────────────────────────────
IMG_SIZE      = 256
TRAIN_SPLIT   = 0.70
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.15
SEED          = 42

# ─── Model ────────────────────────────────────────────────────────────────────
NUM_CLASSES   = 3       # 0=benign, 1=malignant, 2=no-object
NUM_QUERIES   = 100
HIDDEN_DIM    = 256
NHEAD         = 8
ENC_LAYERS    = 3
DEC_LAYERS    = 3
DIM_FFN       = 512

# ─── Training ─────────────────────────────────────────────────────────────────
EPOCHS        = 50
BATCH_SIZE    = 4
LR            = 1e-4    # transformer + heads
LR_BACKBONE   = 1e-5    # pre-trained backbone + projection layers (10x lower)
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 0.1     # max_norm for gradient clipping (standard for DETR)

# ─── Loss ─────────────────────────────────────────────────────────────────────
NO_OBJ_WEIGHT     = 0.1   # down-weight the 99 no-object queries vs 1 real object
PRIOR_LOSS_WEIGHT = 0.5   # geometric prior loss coefficient

# ─── Matcher ──────────────────────────────────────────────────────────────────
COST_CLASS    = 1.0
COST_BBOX     = 5.0

# ─── Evaluation ───────────────────────────────────────────────────────────────
IOU_THRESHOLD = 0.5     # threshold for a detection to count as TP
CLASS_NAMES   = ["benign", "malignant"]
