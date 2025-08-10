"""
Example of use: move 3 bodies and display the collision/distance witnesses.
"""

import time

import numpy as np
import pinocchio as pin

from schaeffler2025.meshcat_viewer_wrapper import MeshcatVisualizer
from tp5.display_witness import DisplayCollisionWitnessesInMeshcat
from tp5.scenes import buildSceneThreeBodies

# %jupyter_snippet build
# Build a scene
model, geom_model = buildSceneThreeBodies()
data = model.createData()
geom_data = geom_model.createData()

# Start meshcat
viz = MeshcatVisualizer(
    model=model, collision_model=geom_model, visual_model=geom_model
)
# %end_jupyter_snippet

# %jupyter_snippet witness
# Build the viewer add-on to display the witnesses.
mcWitnesses = DisplayCollisionWitnessesInMeshcat(viz)
# %end_jupyter_snippet

# Start a random trajectory to display the witnesses.
q = pin.randomConfiguration(model)
# %jupyter_snippet trajectory
v = (np.random.rand(model.nv) * 2 - 1) * 1e-3
r0 = [np.linalg.norm(q[7 * i : 7 * i + 3]) for i in range(model.nq // 7)]
for t in range(100):
    # Update the robot position along an arbitrary trajectory
    q = pin.integrate(model, q, v * 10)
    for i in range(model.nq // 7):
        q[7 * i : 7 * i + 3] *= r0[i] / np.linalg.norm(q[7 * i : 7 * i + 3])
    viz.display(q)

    # Display the witness points
    pin.computeDistances(model, data, geom_model, geom_data, q)
    mcWitnesses.displayDistances(geom_data)

    time.sleep(0.01)
    # %end_jupyter_snippet

    # if HPPFCL3X: # This feature is only implemented with HPPFCL3X
    # %jupyter_snippet distance
    # You can similarly display the witnesses based on distance computation.
    # But if you want to have witnesses when object as far to each other, you
    # need to give some margin.

    # Force the collision margin to a huge value.
    for r in geom_data.collisionRequests:
        r.security_margin = 10

    # Display the witnesses.
    pin.computeCollisions(model, data, geom_model, geom_data, q, False)
    mcWitnesses.displayCollisions(geom_data)
    # %end_jupyter_snippet

assert mcWitnesses.nwitnesses == 3
