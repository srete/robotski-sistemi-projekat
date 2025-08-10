"""
Introduce a solution to display the witness points of a pinocchio
collision request.
The method is only valid for meshcat.
The method only creates visual objects when new collisions appear. Between
two calls, if the number of collision points did not vary, the same visual
objects are kept in meshcat to avoid additional meshcat-internal burdens.
"""

import numpy as np
import pinocchio as pin


class DisplayCollisionWitnessesInMeshcat:
    def __init__(self, viz, point_radius=0.01, linewidth=0.1):
        self.viz = viz
        self.nwitnesses = 0
        self.RADIUS = point_radius
        self.LINEWIDTH = linewidth

        self.A0B0 = np.array([[0, 1], [0, 0], [0, 0], [1, 1]])
        self.lineTransfo = np.eye(4, 4)

    def resetMeshcatObjects(self, nobj):
        """
        Make sure the meshcat objects have been created for <nobj> and no more,
        either by creating the missing objects (ie all at the first iteration, but
        maybe less later) or by removing the extra objects (if more where allocated
        at the previous iteration).
        """
        if nobj < self.nwitnesses:
            # Remove extra objects
            for idx_col in range(nobj, self.nwitnesses):
                self.viz.delete(f"wit_{idx_col}_1")
                self.viz.delete(f"wit_{idx_col}_2")
                self.viz.delete(f"witseg_{idx_col}")
        else:
            for idx_col in range(self.nwitnesses, nobj):
                n = f"wit_{idx_col}_1"
                self.viz.addSphere(n, self.RADIUS, "grey")
                n = f"wit_{idx_col}_2"
                self.viz.addSphere(n, self.RADIUS, "grey")
                n = f"witseg_{idx_col}"
                self.viz.addLine(n, self.A0B0[:3, 0], self.A0B0[:3, 1], "grey")
        self.nwitnesses = nobj

    def _displayOnePair(self, idx_col, p1, p2, normal, dist=None):
        """
        There is a trick for display the witness lines properly.
        In meshcat, lines are best defined with a pair of points.
        Yet meshcat does not give the interface to move these points, but
        only to apply a transform matrix M = [R p].
        We then arbitrarily defines the initial line with points a0,b0 (a0=0,0,0
        and b0=1,0,0 -- using a0=0 but any ||b0||=1 is acceptable).

        Then, if willing to display the line between p1 and p2 (with arbitrary
        coordinates different from a0,b0), we need to choose M=[R p] such that:
        M [a0 b0 ] = [ p1 p2 ] [1 1 ] [ 1 1 ]

        Obviously M=[R p] will not correspond to R being a rotation matrix, but
        this is not a problem for meshcat (indeed, we cannot expect M \in SE3
        to shift a0,b0 to A,B, as the distances are not going to be the same in
        general, so R has to be not a rotation matrix).

        We then chose p=p1 (using a0=0), and R=d.R'
        with d=||p1-p2|| the distance between the witnesses and
        R' the rotation matrix such that R'b0=n=(p2-p1)/d.

        """
        n = f"wit_{idx_col}_1"
        self.viz.applyConfiguration(n, p1.tolist() + [1, 0, 0, 0])

        n = f"wit_{idx_col}_2"
        self.viz.applyConfiguration(n, p2.tolist() + [1, 0, 0, 0])

        n = f"witseg_{idx_col}"
        if dist is None:
            dist = normal @ (p2 - p1)
        quat = pin.Quaternion.FromTwoVectors(self.A0B0[:3, 1], normal)
        self.lineTransfo[:3, 3] = p1
        self.lineTransfo[:3, :3] = quat.matrix() * dist
        self.viz.applyConfiguration(n, self.lineTransfo)

    def displayCollisions(self, geom_data):
        self.resetMeshcatObjects(
            sum([r.numContacts() for r in geom_data.collisionResults])
        )
        idx_col = 0
        for collId, r in enumerate(geom_data.collisionResults):
            if r.numContacts() == 0:
                continue
            for c in r.getContacts():
                assert idx_col < self.nwitnesses
                p1 = c.getNearestPoint1()
                p2 = c.getNearestPoint2()
                self._displayOnePair(idx_col, p1, p2, c.normal)
                idx_col += 1

    def displayDistances(self, geom_data):
        self.resetMeshcatObjects(len(geom_data.distanceResults))
        idx_col = 0
        for collId, r in enumerate(geom_data.distanceResults):
            assert idx_col < self.nwitnesses
            self._displayOnePair(
                idx_col,
                r.getNearestPoint1(),
                r.getNearestPoint2(),
                r.normal,
                r.min_distance,
            )
            idx_col += 1
