from PIL import Image
import imageio 
import os
import datetime

import hppfcl
import pinocchio as pin
import numpy as np
import time
from proxsuite import proxqp
import proxsuite

from tp5.create_rigid_contact_models_for_hppfcl import createContactModelsFromCollisions
from tp5.scenes import buildSceneThreeBodies, buildScenePillsBox, buildSceneCubes, buildSceneRobotHand
from tp5.display_collision_patches import preallocateVisualObjects, updateVisualObjects

from schaeffler2025.meshcat_viewer_wrapper import MeshcatVisualizer

class SimulationConfig:
    def __init__(self, model, dt=1e-4, dt_visu=1/50., duration=5.,
                 mu=0.8, max_staggered_iters=20, staggered_tol=1e-6,
                 enable_friction=True,
                 record_video=False, recording_dir='recordings/', 
                 video_filename=None, video_resolution=(1280, 720)):
        
        # --- Simulation Parameters ---
        self.DT = dt
        self.DT_VISU = dt_visu
        self.DURATION = duration
        self.T = int(self.DURATION / self.DT)
        self.q = model.referenceConfigurations['default']
        self.v = np.zeros(model.nv)
        self.tau = np.zeros(model.nv)
        self.MU = mu
        self.enable_friction = enable_friction
        self.MAX_STAGGERED_ITERS = max_staggered_iters
        self.STAGGERED_TOL = staggered_tol

        # --- Video Recording Parameters ---
        self.record_video = record_video
        self.recording_dir = recording_dir
        if self.record_video:
            if video_filename is None:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.video_filename = f"simulation_{timestamp}.mp4"
            else:
                self.video_filename = video_filename
            
            self.video_filepath = os.path.join(self.recording_dir, self.video_filename)
            self.video_fps = int(1 / self.DT_VISU) if self.DT_VISU > 0 else 30
            self.video_resolution = video_resolution

class Simulation:
    def __init__(self, config, model, data, geom_model, geom_data):
        self.config = config
        self.model = model
        self.data = data
        self.geom_model = geom_model
        self.geom_data = geom_data
        self.contact_models = []
        self.contact_datas = []
        self.writer = None
        
        self.initialize_viz()
        
        if self.config.record_video:
            self.initialize_recording()

    def initialize_viz(self):
        self.visual_model = self.geom_model.copy()
        num_geoms = len(self.geom_model.geometryObjects)
        # Ensure we are accessing the collision request correctly
        if self.geom_data.collisionRequests:
             num_max_contacts = self.geom_data.collisionRequests[0].num_max_contacts
             preallocateVisualObjects(self.visual_model, num_geoms * num_max_contacts)
        self.viz = MeshcatVisualizer(model=self.model, collision_model=self.geom_model, visual_model=self.visual_model)
        updateVisualObjects(self.model, self.data, [], [], self.visual_model, self.viz)

    def initialize_recording(self):
        """Sets up the video writer."""
        if not os.path.exists(self.config.recording_dir):
            os.makedirs(self.config.recording_dir)
        
        print(f"Recording video to {self.config.video_filepath}")
        print(f"  FPS: {self.config.video_fps}, Resolution: {self.config.video_resolution}")
        
        self.writer = imageio.get_writer(self.config.video_filepath, fps=self.config.video_fps)

    def get_staggered_jacobians_from_pinocchio(self):
        '''Extracts the normal and tangential components of the constraints Jacobian'''
        J_full = pin.getConstraintsJacobian(self.model, self.data, self.contact_models, self.contact_datas)
        J_n = J_full[2::3, :]
        nc, nv = len(self.contact_models), self.model.nv
        J_t = np.zeros((4 * nc, nv))
        J_t1, J_t2 = J_full[0::3, :], J_full[1::3, :]
        J_t[0::4, :], J_t[1::4, :], J_t[2::4, :], J_t[3::4, :] = J_t1, -J_t1, J_t2, -J_t2
        E = np.zeros((nc, nc * 4))
        for i in range(nc): E[i, i * 4 : (i + 1) * 4] = 1.0
        return J_n, J_t, E

    def run(self):
        '''Run the simulation loop'''
        
        # Initialize quadratic programming solver
        QP = proxsuite.proxqp.dense.QP

        # Display initial configuration
        self.viz.display(self.config.q)
        self.viz.last_time = time.time()
        time.sleep(1.0)

        # Extract initial configuration
        q, v, tau = self.config.q, self.config.v, self.config.tau
        DT = self.config.DT

        # Main simulation loop
        for t in range(self.config.T):
            # Compute collisions and create contact models
            pin.computeCollisions(self.model, self.data, self.geom_model, self.geom_data, q)
            self.contact_models = createContactModelsFromCollisions(self.model, self.data, self.geom_model, self.geom_data)
            self.contact_datas = [cm.createData() for cm in self.contact_models]
            nc = len(self.contact_models)

            # Compute free dynamics
            pin.computeAllTerms(self.model, self.data, q, v)
            tau.fill(0)
            a_free = pin.aba(self.model, self.data, q, v, tau)
            vf = v + DT * a_free

            # Solve for contact dynamics if there are contacts
            if nc == 0:
                v = vf
            else:
                # First QP -- Solve for normal contact forces
                pin.computeAllTerms(self.model, self.data, q, v)
                J = -pin.getConstraintsJacobian(self.model, self.data, self.contact_models, self.contact_datas)[2::3,:]
                M = self.data.M
                
                qp1 = QP(self.model.nv, 0, nc, False)
                qp1.init(H=M, g=-M @ vf, A=None, b=None, C=J, l=np.zeros(nc))
                qp1.settings.eps_abs = 1e-12
                qp1.solve()
                vf_contact = qp1.results.x
                
                if self.config.enable_friction:
                    # Second QP -- Solve for friction forces
                    # velocity from the first QP is the free velocity for the friction solve
                    J_n, J_t, E = self.get_staggered_jacobians_from_pinocchio()
                    Minv = np.linalg.inv(self.data.M)
                    G_n, G_t, G_nt = J_n @ Minv @ J_n.T, J_t @ Minv @ J_t.T, J_n @ Minv @ J_t.T
                    
                    alpha = np.zeros(nc)
                    beta = np.zeros(J_t.shape[0])

                    for k in range(self.config.MAX_STAGGERED_ITERS):
                        beta_old = beta.copy()
                        
                        # Solve for normal forces (alpha)
                        qp_g_n = J_n @ vf_contact + G_nt @ beta
                        qp_contact = proxqp.dense.QP(nc, 0, nc, False)
                        qp_contact.init(H=G_n, g=qp_g_n, A=None, b=None, C=np.eye(nc), l=np.zeros(nc))
                        qp_contact.solve()
                        alpha = qp_contact.results.x

                        # Solve for tangential forces (beta)
                        qp_g_t = J_t @ vf_contact + G_nt.T @ alpha
                        qp_friction = proxqp.dense.QP(J_t.shape[0], 0, J_t.shape[0] + nc, False)
                        C_friction = np.vstack([E, -np.eye(J_t.shape[0])])
                        u_friction = np.hstack([self.config.MU * alpha, np.zeros(J_t.shape[0])])
                        qp_friction.init(H=G_t, g=qp_g_t, A=None, b=None, C=C_friction, u=u_friction)
                        qp_friction.solve()
                        beta = qp_friction.results.x

                        # Check for convergence
                        if np.linalg.norm(beta - beta_old) < self.config.STAGGERED_TOL: break
                    
                    # Update velocity with contact and friction impulses
                    delta_v = Minv @ (J_n.T @ alpha + J_t.T @ beta)
                    v = vf_contact + delta_v
                else:
                    # If friction is not enabled, just use the contact velocity
                    v = vf_contact

            # Integrate the new velocity to get the next state
            q = pin.integrate(self.model, q, v * DT)

            #  Visualize the simulation at the specified frequency
            if self.config.DT_VISU is not None and (t * DT) % self.config.DT_VISU < DT:
                updateVisualObjects(self.model, self.data, self.contact_models, self.contact_datas, self.visual_model, self.viz)
                self.viz.display(q)
                
                if self.config.record_video and self.writer is not None:
                    img = self.viz.viewer.get_image()
                    img_resized = img.resize(self.config.video_resolution, Image.Resampling.LANCZOS)
                    self.writer.append_data(np.array(img_resized))
                
                time.sleep(max(0, self.config.DT_VISU - (time.time() - self.viz.last_time)))
                self.viz.last_time = time.time()
        
        if self.config.record_video and self.writer is not None:
            self.writer.close()
            print(f"Video saved successfully to {self.config.video_filepath}")


# Test the simulation with a simple example

if __name__ == "__main__":


    # --- MODEL ---
    model, geom_model = buildSceneCubes(3)
    data = model.createData()
    geom_data = geom_model.createData()

    for req in geom_data.collisionRequests:
        req.security_margin = 1e-2
        req.num_max_contacts = 50
        req.enable_contact = True


    # --- SIMULATION PARAMETERS ---
    DT = 1e-4
    DT_VISU = 1/50.
    DURATION = 5.
    MU = 0.8
    MAX_STAGGERED_ITERS = 20
    STAGGERED_TOL = 1e-6
    ENABLE_FRICTION = False

    # Configure the simulation
    config = SimulationConfig(model, dt=DT, dt_visu=DT_VISU, duration=DURATION, 
                              mu=MU, max_staggered_iters=MAX_STAGGERED_ITERS, staggered_tol=STAGGERED_TOL,
                              enable_friction=ENABLE_FRICTION,
                              record_video=True, video_filename="cubes_no_friction.mp4",) 
                              
    simulation = Simulation(config, model, data, geom_model, geom_data)
    
    # Run simulator
    simulation.run()