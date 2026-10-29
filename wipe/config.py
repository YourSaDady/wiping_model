"""Task, controller, and table parameters shared by every component.

All numbers here are design choices recorded in DECISIONS.md. Units are SI.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# ---------------------------------------------------------------- timing
DT_SIM = 1e-3          # MultibodyPlant discrete step [s]
DT_POLICY = 0.05       # policy / expert / logging period (20 Hz), zero-order hold
SUBSTEPS = 5           # coverage samples per policy step (100 Hz)
T_MAX = 50.0           # time limit for learned policies and baselines [s]


# ---------------------------------------------------------------- task
@dataclass(frozen=True)
class TaskParams:
    f_des: float = 10.0        # desired normal force [N]
    f_min: float = 5.0         # "force" coverage: a sample wipes only if f_n >= f_min
    f_max: float = 20.0        # trial fails if the 1 kHz normal force ever exceeds this
    eps_u: float = 0.10        # trial fails if the uncovered fraction exceeds this
    r_wipe: float = 0.02       # virtual wiping-patch radius [m] (task parameter)
    n_lanes: int = 6           # lanes per region; lane spacing s = L / (n_lanes - 1)
    grid_res: float = 0.002    # coverage grid cell size [m]
    # Coverage criterion (README "Success criterion"):
    #   "work"  - a cell is wiped once the friction work done while it was under the patch,
    #             sum over samples of |f_t . v_t| dt, reaches w_min (Archard-style wear dose);
    #   "force" - the proposal's original rule: wiped once under the patch at a sample with f_n >= f_min.
    coverage_mode: str = "work"
    w_min: float = 0.06        # [J] = mu * f_min * 2 r_wipe = 0.3 * 5 N * 4 cm: one full patch-diameter
                               # pass at the minimum force. Calibrated so that the expert path wipes the
                               # region at f_des = 10 N but not at f_n <= f_min (README "Success criterion")


TASK = TaskParams()

# Lane-path speed profile (expert and hybrid baseline): the path speed dips
# linearly to CORNER_SCALE * v within CORNER_DIST (arc length) of every corner.
CORNER_DIST = 0.02
CORNER_SCALE = 0.35

# ---------------------------------------------------------------- tool
BALL_RADIUS = 0.02             # collision sphere radius R [m]
TOOL_OFFSET = 0.17             # ball centre along iiwa_link_7 +z [m]
TOOL_MASS = 0.3                # [kg]
FRICTION = 0.3                 # Coulomb friction (static = dynamic)
HC_DISSIPATION = 20.0          # Hunt-Crossley dissipation [s/m]


# ---------------------------------------------------------------- controller
@dataclass(frozen=True)
class Gains:
    """Cartesian impedance gains. Translational K, D are diagonal in the table
    frame (t1, t2, n): K = K_t I + (K_n - K_t) n n^T, same for D."""
    K_t: float = 1000.0
    K_n: float = 500.0
    D_t: float = 110.0
    D_n: float = 250.0
    K_rot: float = 100.0
    D_rot: float = 10.0
    K_null: float = 5.0
    D_null: float = 1.0

    def K(self, n: np.ndarray) -> np.ndarray:
        n = np.asarray(n, float)
        return self.K_t * np.eye(3) + (self.K_n - self.K_t) * np.outer(n, n)

    def D(self, n: np.ndarray) -> np.ndarray:
        n = np.asarray(n, float)
        return self.D_t * np.eye(3) + (self.D_n - self.D_t) * np.outer(n, n)

    def K_inv(self, n: np.ndarray) -> np.ndarray:
        n = np.asarray(n, float)
        return np.eye(3) / self.K_t + (1.0 / self.K_n - 1.0 / self.K_t) * np.outer(n, n)

    @property
    def tag(self) -> str:
        return f"K{int(self.K_n)}"


MAIN_GAINS = Gains()
GAINS_BY_KN = {250: replace(MAIN_GAINS, K_n=250.0),
               500: MAIN_GAINS,
               1000: replace(MAIN_GAINS, K_n=1000.0)}


# ---------------------------------------------------------------- tables
@dataclass(frozen=True)
class TableSpec:
    name: str
    tilt_deg: float
    k_e: float                 # effective normal contact stiffness [N/m]
    split: str = "train"       # "train" (in-distribution) or "ood"

    @property
    def normal(self) -> np.ndarray:
        """Table normal: tilt is a rotation about world +x."""
        th = np.deg2rad(self.tilt_deg)
        return np.array([0.0, -np.sin(th), np.cos(th)])


def _ke_tag(k: float) -> str:
    if k < 1e3:
        return f"{int(k)}"
    e = int(np.floor(np.log10(k)))
    m = k / 10 ** e
    return f"{m:g}e{e}"


def _table(tilt, k_e, split="train"):
    return TableSpec(f"tilt{int(tilt):02d}_ke{_ke_tag(k_e)}", float(tilt), float(k_e), split)


TRAIN_TILTS = (0, 10, 20)
TRAIN_KE = (5e2, 2e3, 2e4, 1e5)
TRAIN_TABLES = [_table(t, k) for t in TRAIN_TILTS for k in TRAIN_KE]
OOD_TABLES = [
    _table(15, 5e4, "ood"),    # unseen tilt and unseen k_e, both inside the training range
    _table(28, 2e4, "ood"),    # steeper than any training tilt
    _table(28, 5e2, "ood"),    # steeper and soft
    _table(0, 2.5e2, "ood"),   # softer than any training surface
    _table(10, 3e5, "ood"),    # harder than any training surface
]
ALL_TABLES = TRAIN_TABLES + OOD_TABLES
TABLES = {t.name: t for t in ALL_TABLES}

# ---------------------------------------------------------------- dataset sizes
N_DEMOS_PER_TABLE = 36
N_VAL_SCENES_PER_TABLE = 10
N_TEST_SCENES_PER_TABLE = 50
SPLIT_CODE = {"demo": 0, "val": 1, "test": 2, "dev": 3}
