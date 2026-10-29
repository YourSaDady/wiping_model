import os, sys, time
import numpy as np
from wipe.scene import make_scene
from wipe.sim import WipeEnv
from wipe.expert import ScriptedExpert, PHASE_ID
from wipe.config import MAIN_GAINS
table = sys.argv[1] if len(sys.argv) > 1 else "tilt00_ke2e4"
idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sc = make_scene(table, "dev", idx)
t0 = time.time()
env = WipeEnv(sc, MAIN_GAINS)
t1 = time.time()
ex = ScriptedExpert(sc, MAIN_GAINS)
obs = env.observe()
log = []
while not ex.done and env.k < 900:
    xd = ex.act(obs)
    fn = float(sc.arr("n") @ obs["f"])
    log.append((obs["t"], PHASE_ID[ex.phase], fn, sc.to_plane(obs["x"])[2] - 0.02, *sc.to_plane(xd)))
    obs = env.step(xd)
t2 = time.time()
m = env.metrics()
print(f"build {t1-t0:.2f}s sim {t2-t1:.2f}s for {env.k*0.05:.1f}s sim; v={sc.v:.3f} L={sc.L:.3f}")
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
log = np.array(log)
for ph in range(5):
    s = log[log[:, 1] == ph]
    if len(s): print(ph, len(s), "fn mean %.2f min %.2f max %.2f" % (s[:, 2].mean(), s[:, 2].min(), s[:, 2].max()), "gap min %.4f" % s[:, 3].min())
os.makedirs("results/milestone", exist_ok=True)
np.save("results/milestone/try_expert_log.npy", log)   # (t, phase, f_n, gap, x_d in region frame)
