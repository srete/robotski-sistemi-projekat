import time
import unittest

import numpy as np
import pinocchio as pin

from schaeffler2025.meshcat_viewer_wrapper import MeshcatVisualizer
from tp5.scenes import buildSceneRobotHand

# %jupyter_snippet robothand
model, geom_model = buildSceneRobotHand()
data = model.createData()
visual_model = geom_model.copy()
viz = MeshcatVisualizer(
    model=model, collision_model=geom_model, visual_model=visual_model
)
q0 = model.referenceConfigurations["default"]
viz.display(q0)
# %end_jupyter_snippet

# Initial state position+velocity
# %jupyter_snippet init
q = q0.copy()
vq = np.zeros(model.nv)
# %end_jupyter_snippet

# %jupyter_snippet hyper
# Hyperparameters for the simu
DT = 1e-3  # simulation timestep
DT_VISU = 1 / 50
DURATION = 3.0  # duration of simulation
T = int(DURATION / DT)  # number of time steps
# %end_jupyter_snippet
# %jupyter_snippet hyper_control
# Hyperparameters for the control
Kp = 50.0  # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)  # derivative gain (D of PD)
# %end_jupyter_snippet

### Examples for computing the generalized mass matrix and dynamic bias.
# %jupyter_snippet mass
M = pin.crba(model, data, q)
b = pin.nle(model, data, q, vq)
# %end_jupyter_snippet

### Example to compute the forward dynamics from M and b
# %jupyter_snippet dyninv
tauq = np.random.rand(model.nv) * 2 - 1
aq = np.linalg.inv(M) @ (tauq - b)
# %end_jupyter_snippet
# Alternatively, call the ABA algorithm
aq_bis = pin.aba(model, data, q, vq, tauq)
print(f"Sanity check, should be 0 ... {np.linalg.norm(aq-aq_bis)}")
assert np.allclose(aq, aq_bis)

### Example to integrate an acceleration.
# %jupyter_snippet integrate
vq += aq * DT
q = pin.integrate(model, q, vq * DT)
# %end_jupyter_snippet

### Reference trajectory
# %jupyter_snippet trajref
from tp5.traj_ref import TrajRef  # noqa E402

qdes = TrajRef(
    q0,
    omega=np.array([0, 0.1, 1, 1.5, 2.5, -1, -1.5, -2.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    amplitude=1.5,
)
# %end_jupyter_snippet

# %jupyter_snippet loop
hq = []  ### For storing the logs of measured trajectory q
hqdes = []  ### For storing the logs of desired trajectory qdes
for i in range(T):
    t = i * DT

    # Compute the model.
    M = pin.crba(model, data, q)
    b = pin.nle(model, data, q, vq)

    # Compute the PD control.
    tauq = -Kp * (q - qdes(t)) - Kv * (vq - qdes.velocity(t)) + qdes.acceleration(t)

    # Simulated the resulting acceleration (forward dynamics
    aq = np.linalg.inv(M) @ (tauq - b)

    # Integrate the acceleration.
    vq += aq * DT
    q = pin.integrate(model, q, vq * DT)

    # Display once in a while...
    if DT_VISU is not None and abs((t) % DT_VISU) <= 0.9 * DT:
        viz.display(q)
        time.sleep(DT_VISU)

    # Log the history.
    hq.append(q.copy())
    hqdes.append(qdes.copy())

# %end_jupyter_snippet


### TEST ZONE ############################################################
### This last part is to automatically validate the versions of this example.
class PDTest(unittest.TestCase):
    def test_logs(self):
        print(self.__class__.__name__)
        model, gmodel = buildSceneRobotHand()
        data = model.createData()
        q = model.referenceConfigurations["default"].copy()
        vq = np.random.rand(model.nv)
        tauq = np.random.rand(model.nv)

        M = pin.crba(model, data, q)
        b = pin.nle(model, data, q, vq)
        aq = np.linalg.inv(M) @ (tauq - b)
        aq_bis = pin.aba(model, data, q, vq, tauq)
        self.assertTrue(np.allclose(aq, aq_bis))
        self.assertTrue(len(hq) == len(hqdes))


if __name__ == "__main__":
    PDTest().test_logs()
