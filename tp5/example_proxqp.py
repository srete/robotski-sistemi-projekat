"""
Example of use of the QP solver ProxQP.
See reference paper: Bambade et al, RSS 2022, https://www.roboticsproceedings.org/rss18/p040.pdf

The example randomly sample a QP problem with NX variables, NEQ equality constraints,
NIEQ inequality constraints and optional bounds (box constraints).
Then, the solution is checked through various primal and dual tests.
The thresholds of the tests could be more accurately adjusted, so some random instances
might fail the test due to small unaccuracies.

decide:
   x in R^NX

minimizing:
   .5 xHx + gx

subject to:
   bounds[:,0] <= x <= bounds[:,1] (multiplier denoted by w)
   Ae x = be    (multipler denoted by y)
   bi[:,0] <= Ai x <= bi[:,1] (multiplier denoted by z)

"""

import random
import time

import numpy as np
import proxsuite
from numpy.linalg import eig

SEED = 1470
SEED = int(time.time() % 1 * 10000)
print("SEED = ", SEED)
np.random.seed(SEED)
random.seed(SEED)

# %jupyter_snippet param
# ### TEST PARAMETERS
NX = 20  # x dimension (search space)
NEQ = 5  # number of equalities
NINEQ = 3  # Number of inequalities
WITH_BOUNDS = True  # Additional bounds on x
VERBOSE = True  # Do you want to see the result?
ACCURACY = 1e-6  # Threshold for solver stoping criteria and posterior checks
# %end_jupyter_snippet

# ### PROBLEM SETUP
# %jupyter_snippet matrices

# Cost
H = np.random.rand(NX, NX) * 2 - 1
H = H @ H.T  ### Make it positive symmetric
g = np.random.rand(NX)

Ae = np.random.rand(NEQ, NX) * 2 - 1
be = np.random.rand(NEQ) * 2 - 1

Ai = np.random.rand(NINEQ, NX) * 2 - 1
bi = np.sort(np.random.rand(NINEQ, 2) * 2 - 1, 1)
for i in range(NINEQ):
    # Half inequalities are double bounds
    # One quarter are pure lower
    # One quarter are pure upper
    r = random.randint(0, 3)
    if r == 0:
        bi[i, 0] = -1e20
    elif r == 1:
        bi[i, 1] = 1e20

bounds = np.sort(np.random.rand(NX, 2) * 2 - 1, 1) + [-1, 1]
for i in range(NX):
    # Half inequalities are double bounds
    # One quarter are pure lower
    # One quarter are pure upper
    r = random.randint(0, 3)
    if r == 0:
        bounds[i, 0] = -1e20
    elif r == 1:
        bounds[i, 1] = 1e20
# %end_jupyter_snippet

assert np.all(eig(H)[0] > 0)
assert np.all(bi[:, 0] < bi[:, 1])
assert np.all(bounds[:, 0] < bounds[:, 1])

# ### SOLVER SETUP
# %jupyter_snippet solve
# [x, cost, _, niter, lag, iact] =
qp = proxsuite.proxqp.dense.QP(NX, NEQ, NINEQ, WITH_BOUNDS)
qp.settings.eps_abs = ACCURACY / 1e3
qp.init(
    H,
    g,
    Ae,
    be,
    Ai,
    bi[:, 0],
    bi[:, 1],
    bounds[:, 0] if WITH_BOUNDS else None,
    bounds[:, 1] if WITH_BOUNDS else None,
)
qp.solve()
# %end_jupyter_snippet

# ### RESULT
# %jupyter_snippet result
x, y, z = qp.results.x, qp.results.y, qp.results.z
if WITH_BOUNDS:
    w = z[NINEQ:]  # bounds
    z = z[:NINEQ]  # general inequalities
cost = qp.results.info.objValue
# %end_jupyter_snippet

# ### VERBOSE
if VERBOSE:
    # print an optimal solution
    # %jupyter_snippet print
    print("Primal optimum x: {}".format(x))
    print("Dual optimum (equalities) y: {}".format(y))
    print("Dual optimum (ineq) z: {}".format(z))
    print("Dual optimum (bounds) w: {}".format(w))
    # %end_jupyter_snippet

# ### CHECK THE RESULT
# Sanity check the obtained cost.
assert np.isclose(x @ H @ x / 2 + g @ x, cost)

# Check primal KKT condition
assert np.allclose(Ae @ x, be)
assert np.all(Ai @ x >= bi[:, 0] - ACCURACY)
assert np.all(Ai @ x <= bi[:, 1] + ACCURACY)

# Check complementarity
assert abs(z @ np.min([(Ai @ x - bi[:, 0]), (-Ai @ x + bi[:, 1])], 0)) < ACCURACY
assert abs(w @ np.min([(x - bounds[:, 0]), (-x + bounds[:, 1])], 0)) < ACCURACY
# Check complementarity of inequality side
for i in range(NINEQ):
    if abs(z[i]) > ACCURACY * 1e3:
        # z negative corresponds to lower bound reach
        # z positive corresponds to upper bound reach
        assert (z[i] < 0 and abs(Ai[i] @ x - bi[i, 0]) < ACCURACY) or (
            z[i] > 0 and abs(Ai[i] @ x - bi[i, 1]) < ACCURACY
        )
# Check complementarity of bound side
for i in range(NX):
    if abs(w[i]) > ACCURACY * 1e3:
        # w negative corresponds to lower bound reach
        # w positive corresponds to upper bound reach
        assert (w[i] < 0 and abs(x[i] - bounds[i, 0]) < ACCURACY) or (
            w[i] > 0 and abs(x[i] - bounds[i, 1]) < ACCURACY
        )
