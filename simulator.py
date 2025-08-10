# --- NEW: Import Image from Pillow for resizing ---
from PIL import Image
import imageio 

# (The rest of your imports are the same)
import hppfcl
import pinocchio as pin
import numpy as np
import time
from proxsuite import proxqp
import proxsuite
QP = proxsuite.proxqp.dense.QP

from tp5.create_rigid_contact_models_for_hppfcl import createContactModelsFromCollisions
from tp5.scenes import buildSceneThreeBodies, buildScenePillsBox, buildSceneCubes, buildSceneRobotHand

from tp5.display_collision_patches import preallocateVisualObjects, updateVisualObjects
from schaeffler2025.meshcat_viewer_wrapper import MeshcatVisualizer

# (Your utility function remains the same)
def get_staggered_jacobians_from_pinocchio(model, data, contact_models, contact_datas):
    J_full = pin.getConstraintsJacobian(model, data, contact_models, contact_datas)
    J_n = J_full[2::3, :]
    nc, nv = len(contact_models), model.nv
    J_t = np.zeros((4 * nc, nv))
    J_t1, J_t2 = J_full[0::3, :], J_full[1::3, :]
    J_t[0::4, :], J_t[1::4, :], J_t[2::4, :], J_t[3::4, :] = J_t1, -J_t1, J_t2, -J_t2
    E = np.zeros((nc, nc * 4))
    for i in range(nc): E[i, i * 4 : (i + 1) * 4] = 1.0
    return J_n, J_t, E

# --- SIMULATION SETUP ---
model, geom_model = buildSceneCubes(3)
data = model.createData()
geom_data = geom_model.createData()

for req in geom_data.collisionRequests:
    req.security_margin = 1e-2
    req.num_max_contacts = 50
    req.enable_contact = True

# --- VIZUALIZATION ---
visual_model = geom_model.copy()
num_geoms = len(geom_model.geometryObjects)
preallocateVisualObjects(visual_model, num_geoms * req.num_max_contacts)
viz = MeshcatVisualizer(model=model, collision_model=geom_model, visual_model=visual_model)
updateVisualObjects(model,data,[],[],visual_model,viz)

# --- SIMULATION PARAMETERS ---
DT = 1e-4
DT_VISU = 1/50.
DURATION = 5.
T = int(DURATION / DT)
q = model.referenceConfigurations['default']
v = np.zeros(model.nv)
tau = np.zeros(model.nv)

MU = 0.8
MAX_STAGGERED_ITERS = 20
STAGGERED_TOL = 1e-6

# --- VIDEO RECORDING SETUP ---
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
VIDEO_FILENAME = f"cubes_simulation_{timestamp}.mp4"
VIDEO_FPS = int(1 / DT_VISU)
# --- NEW: Define a fixed, standard resolution for the video ---
# These dimensions are divisible by 16, which is ideal for most video codecs.
VIDEO_RESOLUTION = (1280, 720) 

# writer = imageio.get_writer(VIDEO_FILENAME, fps=VIDEO_FPS)
print(f"Recording video to {VIDEO_FILENAME} at {VIDEO_FPS} FPS.")
print(f"Video resolution will be fixed to {VIDEO_RESOLUTION}.")


# --- SIMULATION LOOP ---
viz.display(q)
viz.last_time = time.time()
time.sleep(1.0) 

for t in range(T):
    pin.computeCollisions(model, data, geom_model, geom_data, q)
    contact_models = createContactModelsFromCollisions(model, data, geom_model, geom_data)
    contact_datas = [cm.createData() for cm in contact_models]
    nc = len(contact_models)
 
    pin.computeAllTerms(model, data, q, v)
    tau.fill(0)
    a_free = pin.aba(model, data, q, v, tau)
    vf = v + DT * a_free

    if nc == 0:
        v = vf
    else:
        pin.computeAllTerms(model, data, q, v)
        J = -pin.getConstraintsJacobian(model, data, contact_models, contact_datas)[2::3,:]
        assert(J.shape == (nc,model.nv))
        M = data.M
        
        qp1 = QP(model.nv, 0, nc, False)
        qp1.init(H=data.M, g=-data.M @ vf, A=None, b=None, C=J, l=np.zeros(nc))
        qp1.settings.eps_abs = 1e-12
        qp1.solve()
        vnext = qp1.results.x
        
        J_n, J_t, E = get_staggered_jacobians_from_pinocchio(model, data, contact_models, contact_datas)
        Minv = np.linalg.inv(data.M)
        G_n, G_t, G_nt = J_n @ Minv @ J_n.T, J_t @ Minv @ J_t.T, J_n @ Minv @ J_t.T
        alpha = np.zeros(nc)
        beta = np.zeros(J_t.shape[0])
        
        vf = vnext

        for k in range(MAX_STAGGERED_ITERS):
            beta_old = beta.copy()
            qp_g_n = J_n @ vf + G_nt @ beta
            qp_contact = proxqp.dense.QP(nc, 0, nc, False)
            qp_contact.init(H=G_n, g=qp_g_n, A=None, b=None, C=np.eye(nc), l=np.zeros(nc))
            qp_contact.solve()
            alpha = qp_contact.results.x

            qp_g_t = J_t @ vf + G_nt.T @ alpha
            qp_friction = proxqp.dense.QP(J_t.shape[0], 0, J_t.shape[0] + nc, False)
            C_friction = np.vstack([E, -np.eye(J_t.shape[0])])
            u_friction = np.hstack([MU * alpha, np.zeros(J_t.shape[0])])
            qp_friction.init(H=G_t, g=qp_g_t, A=None, b=None, C=C_friction, u=u_friction)
            qp_friction.solve()
            beta = qp_friction.results.x

            if np.linalg.norm(beta - beta_old) < STAGGERED_TOL: break
        
        delta_v = Minv @ (J_n.T @ alpha + J_t.T @ beta)
        v = vf + delta_v

    q = pin.integrate(model, q, v * DT)

    if DT_VISU is not None and (t*DT) % DT_VISU < DT:
        updateVisualObjects(model, data, contact_models, contact_datas, visual_model, viz)
        viz.display(q)
        
        # --- FRAME CAPTURE AND RESIZING ---
        # Capture the image from the viewer
        img = viz.viewer.get_image()
        
        # --- NEW: Resize the image to our fixed resolution ---
        # This ensures every frame has the same size, fixing the ValueError.
        # We use LANCZOS for high-quality resampling.
        img_resized = img.resize(VIDEO_RESOLUTION, Image.Resampling.LANCZOS)
        
        # Add the *resized* image (as a numpy array) to the video writer
        # writer.append_data(np.array(img_resized))
        
        time.sleep(max(0, DT_VISU - (time.time() - viz.last_time)))
        viz.last_time = time.time()

# --- FINALIZE AND SAVE VIDEO ---
 #writer.close()
print(f"Video saved successfully to {VIDEO_FILENAME}")