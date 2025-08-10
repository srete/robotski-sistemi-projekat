import random
import unittest
import warnings

import hppfcl
import numpy as np
import pinocchio as pin

from tp5.robot_hand import RobotHand
from example_robot_data.robots_loader import load as load_robot

def buildScenePillsBox(
    nobj=30, wall_size=4.0, seed=0, no_walls=False, one_of_each=False
):
    """
    Create pinocchio models (pin.Model and pin.GeometryModel) for a scene
    composed of a box (6 walls partly transparent) and <nobj> objects of small
    size (capsules, ellipsoid, convex patatoid from STL).
    The scene can be varied by changing the number of objects <nobj>, the size of the
    box <wall_size> and the random seed <seed>.
    If no_walls is True, then don't build the walls.
    If one_of_each is True, then don't randomize the type of the objects.

    Parameters:
    - nobj (int): The number of objects to generate (default is 30).
    - wall_size (float): The size of the walls of the box (default is 4.0).
    - seed (int): Seed value for random number generation (default is 0).
    - no_walls (bool): If True, no box is added (default is False).
    - one_of_each (bool): If True, only one object of each type (ellipsoid,
    capsule, weird-shape) will be generated in turn (1,2,3,1,2,3,...) (default
    is False).

    Returns:
    - scene: A generated scene containing the specified number of pill-shaped
    objects within a box.

    """
    SEED = seed
    NOBJ = nobj
    WORLD_SIZE = 0.45 * wall_size

    # Initialize the random generators
    pin.seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    model = pin.Model()
    geom_model = pin.GeometryModel()

    # ###
    # ### OBJECT SAMPLING
    # ###
    # Sample objects with the following classes.
    shapes = [
        # load_hppfcl_convex_from_stl_file("schaeffler2025/share/mesh.stl"),
        hppfcl.Ellipsoid(0.05, 0.15, 0.2),
        hppfcl.Capsule(0.1, 0.2),
    ]

    for s in shapes:
        s.computeLocalAABB()

    # Joint limits
    world_bounds = np.array([WORLD_SIZE] * 3 + [np.inf] * 4)

    if one_of_each:
        shapeSamples = []
        for _ in range((nobj + 1) // len(shapes)):
            shapeSamples.extend(shapes)
        shapeSamples = shapeSamples[:nobj]
    else:
        shapeSamples = random.sample(shapes * NOBJ, k=NOBJ)

    for shape in shapeSamples:
        jid = model.addJoint(
            0,
            pin.JointModelFreeFlyer(),
            pin.SE3.Identity(),
            "obj1",
            min_config=-world_bounds,
            max_config=world_bounds,
            max_velocity=np.ones(6),
            max_effort=np.ones(6),
        )
        color = np.random.rand(4)
        color[-1] = 1
        # Place the object so that it is centered
        Mcenter = pin.SE3(np.eye(3), -shape.computeCOM())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geom = pin.GeometryObject(
                f"{str(type(shape))[22:-2]}_{jid}",
                jid,
                jid,
                placement=Mcenter,
                collision_geometry=shape,
            )

        geom.meshColor = np.array(color)
        geom_model.addGeometryObject(geom)
        # Add inertia
        volum = shape.computeVolume()
        I3 = shape.computeMomentofInertia()
        density = 700  # kg/m3 = wood
        model.appendBodyToJoint(
            jid,
            pin.Inertia(volum * density, np.zeros(3), I3 * density),
            pin.SE3.Identity(),
        )

    # Add all pairs
    geom_model.addAllCollisionPairs()

    if not no_walls:
        addBox(geom_model, wall_size)

    return model, geom_model


def buildSceneThreeBodies(seed=0, size=1.0):
    """
    Build a new scenes composed of 3 floating objects.
    This function is a proxy over the more general buildScenePillsBox.

    Parameters:
    - seed (int): Seed value for random number generation (default is 0).
    - size (float): The size of the scene (default is 1.0).

    Returns:
    - scene: A generated scene containing three pill-shape floating objects.
    """
    return buildScenePillsBox(
        nobj=3, no_walls=True, seed=seed, one_of_each=True, wall_size=size
    )


def addFloor(
    geom_model,
    altitude=0,
    addCollisionPairs=True,
    color=np.array([0.9, 0.6, 0.0, 0.20]),
):
    """
    Add an infinite horizontal plan to an existing scene.

    Parameters:
    - geom_model: The geometric model of the scene to which the floor will be added.
    - altitude (float): The altitude of the floor (default is 0).
    - addCollisionPairs (bool): If True, collision pairs will be added for the
    floor with all the other objects already in the scene (default is True).
    - color (numpy.ndarray): The color of the floor in RGBA format (default is
    np.array([0.9, 0.6, 0., .20])) with RGB-transparency syntax.

    """
    shape = hppfcl.Halfspace(np.array([0, 0, 1.0]), 0)
    M = pin.SE3.Identity()
    M.translation[2] = altitude
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        floor = pin.GeometryObject("floor", 0, 0, placement=M, collision_geometry=shape)
    floor.meshColor = color
    ifloor = geom_model.addGeometryObject(floor)

    # Collision pairs between all objects and the floor.
    for ig, g in enumerate(geom_model.geometryObjects):
        if g.name != "floor":
            geom_model.addCollisionPair(pin.CollisionPair(ig, ifloor))


def addBox(
    geom_model, wall_size=4.0, color=np.array([1, 1, 1, 0.2]), transparency=None
):
    """
    Add a box composed of 6 transparent walls forming a cube.
    This box is typically though to come outside of a previously defined object set.

    Parameters:

    - geom_model: The geometric model of the scene to which the box will be added.
    - wall_size (float): The size of each wall of the box (default is 4.0).
    - color (numpy.ndarray): The color of the box walls in RGBA format (default
      is np.array([1, 1, 1, 0.2])).
    - transparency (float or None): The transparency of the box walls (default
    is None). If both color and transparency are set, the last component of the
    color will be ignored.

    """
    WALL_SIZE = wall_size
    WALL_THICKNESS = WALL_SIZE * 0.05

    wall_color = color.copy()
    assert len(wall_color) == 4
    if transparency is not None:
        wall_color[3] = transparency

    # X-axis
    M = pin.SE3.Identity()
    M.translation = np.array([-WALL_SIZE - WALL_THICKNESS, 0.0, 0.0]) / 2
    shape = hppfcl.Box(WALL_THICKNESS, WALL_SIZE, WALL_SIZE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geom = pin.GeometryObject(
            "wall_X-", 0, 0, placement=M, collision_geometry=shape
        )
    geom.meshColor = np.array(wall_color)
    geom_model.addGeometryObject(geom)

    M = pin.SE3.Identity()
    M.translation = np.array([WALL_SIZE, 0.0, 0.0]) / 2
    shape = hppfcl.Box(WALL_THICKNESS, WALL_SIZE, WALL_SIZE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geom = pin.GeometryObject(
            "wall_X+", 0, 0, placement=M, collision_geometry=shape
        )
    geom.meshColor = np.array(wall_color)
    geom_model.addGeometryObject(geom)

    # Y-axis
    M = pin.SE3.Identity()
    M.translation = np.array([0.0, -WALL_SIZE, 0.0]) / 2
    shape = hppfcl.Box(WALL_SIZE, WALL_THICKNESS, WALL_SIZE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geom = pin.GeometryObject(
            "wall_Y-", 0, 0, placement=M, collision_geometry=shape
        )
    geom.meshColor = np.array(wall_color)
    geom_model.addGeometryObject(geom)

    M = pin.SE3.Identity()
    M.translation = np.array([0.0, WALL_SIZE, 0.0]) / 2
    shape = hppfcl.Box(WALL_SIZE, WALL_THICKNESS, WALL_SIZE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geom = pin.GeometryObject(
            "wall_Y+", 0, 0, placement=M, collision_geometry=shape
        )
    geom.meshColor = np.array(wall_color)
    geom_model.addGeometryObject(geom)

    # Z-axis
    M = pin.SE3.Identity()
    M.translation = np.array([0.0, 0.0, -WALL_SIZE]) / 2
    shape = hppfcl.Box(WALL_SIZE, WALL_SIZE, WALL_THICKNESS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geom = pin.GeometryObject(
            "wall_Z-", 0, 0, placement=M, collision_geometry=shape
        )
    geom.meshColor = np.array(wall_color)
    geom_model.addGeometryObject(geom)

    M = pin.SE3.Identity()
    M.translation = np.array([0.0, 0.0, WALL_SIZE]) / 2
    shape = hppfcl.Box(WALL_SIZE, WALL_SIZE, WALL_THICKNESS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geom = pin.GeometryObject(
            "wall_Z+", 0, 0, placement=M, collision_geometry=shape
        )
    geom.meshColor = np.array(wall_color)
    geom_model.addGeometryObject(geom)

    # Remove pairs between walls
    for ig1, g1 in enumerate(geom_model.geometryObjects):
        for ig2, g2 in enumerate(geom_model.geometryObjects):
            # a^b is XOR ( (a and not b) or (b and not a) )
            if ig1 < ig2 and (("wall" in g1.name) ^ ("wall" in g2.name)):
                geom_model.addCollisionPair(pin.CollisionPair(ig1, ig2))


def buildSceneCubes(
    number_of_cubes,
    sizes=0.2,
    masses=1.0,
    with_corner_collisions=False,
    with_cube_collisions=True,
    with_floor=True,
    corner_inside_the_cube=True,
):
    """
    Creates the pinocchio pin.Model and pin.GeometryModel of a scene containing
    cubes. The number of cubes is defines by the length of the two lists in
    argument.  The cubes can be augmented with 8 small spheres corresponding to
    the corners.  In that case, the collisions with the cube geometries are
    disable, and only the collisions between sphere is active.

    Args:

        number_of_cubes: number of cubes (then expect lists of proper sizes)

        sizes: a list containing the lengthes of the cubes. If a single scalar
        is given, then extrapolate it to a list of proper size. Defaults to .2

        masses: a list containing the masses of the cubes. If a single scalar
        is given, then extrapolate it to a list of proper size. Defaults to 1

    option args:

        with_corner_collisions: add balls to the corner of the cube and only
        enable collision checks between these additional balls. True by
        default.

        with_cube_collisions: add collision pairs between cubes. False by
        defaults (why?).

        corner_inside_the_cube: added for compatibility with previous
        code. True by default.

    Returns:
        model, geom_model

    """
    N = number_of_cubes
    if np.isscalar(sizes):
        sizes = [sizes] * N
    if np.isscalar(masses):
        masses = [masses] * N
    assert len(sizes) == len(masses) == N

    BALL_FACTOR_SIZE = 1 / 50
    SPHERE_TEMPLATE_NAME = "cube_corner:{n_cube}:{n_sphere}"
    CUBE_TEMPLATE_NAME = "cube:{n_cube}"
    CUBE_COLOR = np.array([0.0, 0.0, 1.0, 0.6])
    SPHERE_COLOR = np.array([1.0, 0.2, 0.2, 1.0])
    WORLD_BOUNDS = np.array([max(sizes) * 5] * 3 + [np.inf] * 4)

    model = pin.Model()
    geom_model = pin.GeometryModel()

    for n_cube, (size, mass) in enumerate(zip(sizes, masses)):
        # to get random init above the floor
        low = -WORLD_BOUNDS
        low[2] = size * np.sqrt(3) / 2
        jointCube = model.addJoint(
            0,
            pin.JointModelFreeFlyer(),
            pin.SE3.Identity(),
            "joint1_" + str(n_cube),
            min_config=low,
            max_config=WORLD_BOUNDS,
            max_velocity=np.ones(6),
            max_effort=np.ones(6),
        )
        model.appendBodyToJoint(
            jointCube, pin.Inertia.FromBox(mass, size, size, size), pin.SE3.Identity()
        )

        # Add cube visual
        shape = hppfcl.Box(size, size, size)
        print(shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geom_box = pin.GeometryObject(
                CUBE_TEMPLATE_NAME.format(n_cube=n_cube),
                jointCube,
                jointCube,
                placement=pin.SE3.Identity(),
                collision_geometry=shape,
            )

        geom_box.meshColor = CUBE_COLOR
        geom_model.addGeometryObject(geom_box)  # only for visualisation

        # Add corner collisions
        # For each corner at +size/2 or -size/2 for x y and z, add a small sphere.
        ballSize = size * BALL_FACTOR_SIZE
        shape = hppfcl.Sphere(ballSize)
        M = pin.SE3.Identity()
        # TODO: Order x,y,z temporarily chosen to mimic the initial code.
        # Could be reset to a more classical order to avoid confusion later.
        corners = [
            np.array([x, y, z])
            * (size / 2 - (ballSize if corner_inside_the_cube else 0))
            for x in [1, -1]
            for z in [-1, 1]
            for y in [-1, 1]
        ]
        for n_sphere, trans in enumerate(corners):
            M.translation = trans
            name = SPHERE_TEMPLATE_NAME.format(n_cube=n_cube, n_sphere=n_sphere)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                geom_ball1 = pin.GeometryObject(
                    name, jointCube, jointCube, placement=M, collision_geometry=shape
                )
            geom_ball1.meshColor = SPHERE_COLOR
            geom_model.addGeometryObject(geom_ball1)

    # Collision pairs
    if with_corner_collisions:
        # Add collisions between corners of different cubes
        assert "cube_corner:" in SPHERE_TEMPLATE_NAME
        for ig1, g1 in enumerate(geom_model.geometryObjects):
            for ig2, g2 in enumerate(geom_model.geometryObjects):
                if not ig1 < ig2:
                    continue
                n1 = g1.name.split(":")
                n2 = g2.name.split(":")
                if n1[0] == "cube_corner" and n2[0] == "cube_corner":
                    assert len(n1) == 3
                    assert len(n2) == 3
                    if n1[1] != n2[1]:
                        geom_model.addCollisionPair(pin.CollisionPair(ig1, ig2))

    if with_cube_collisions:
        # Add collisions between each cube
        assert "cube:" in CUBE_TEMPLATE_NAME
        for ig1, g1 in enumerate(geom_model.geometryObjects):
            for ig2, g2 in enumerate(geom_model.geometryObjects):
                if not ig1 < ig2:
                    continue
                n1 = g1.name.split(":")
                n2 = g2.name.split(":")
                if n1[0] == "cube" and n2[0] == "cube":
                    assert len(n1) == 2
                    assert len(n2) == 2
                    assert n1[1] != n2[1]
                    geom_model.addCollisionPair(pin.CollisionPair(ig1, ig2))

    if with_cube_collisions and with_corner_collisions:
        # Add collisions between the corners of different cubes
        assert "cube_corner:" in SPHERE_TEMPLATE_NAME
        assert "cube:" in CUBE_TEMPLATE_NAME
        for ig1, g1 in enumerate(geom_model.geometryObjects):
            for ig2, g2 in enumerate(geom_model.geometryObjects):
                if not ig1 < ig2:
                    continue
                n1 = g1.name.split(":")
                n2 = g2.name.split(":")
                if n1[0] == "cube" and n2[0] == "cube_corner":
                    assert len(n1) == 2
                    assert len(n2) == 3
                    if n1[1] != n2[1]:
                        geom_model.addCollisionPair(pin.CollisionPair(ig1, ig2))

    if with_floor:
        addFloor(geom_model, altitude=0)

        # Remove collision between floor and either cube or cube corners.
        for p in [p.copy() for p in geom_model.collisionPairs]:
            pairName = (
                geom_model.geometryObjects[p.first].name
                + geom_model.geometryObjects[p.second].name
            )
            if (
                not with_corner_collisions
                and "floor" in pairName
                and "cube_corner" in pairName
            ):
                geom_model.removeCollisionPair(p)
            if not with_cube_collisions and "floor" in pairName and "cube:" in pairName:
                geom_model.removeCollisionPair(p)

    # Reference configuration
    xy = [(np.random.rand(2) * 2 - 1) * s / 4 for s in sizes]
    z = [[sum(sizes[: i + 1]) * 1.75] for i in range(len(sizes))]
    rot = [pin.Quaternion(pin.SE3.Random().rotation).coeffs() for _ in sizes]
    model.referenceConfigurations["default"] = np.concatenate(
        [np.concatenate(qi) for qi in zip(xy, z, rot)]
    )

    return model, geom_model
def buildSceneHouseOfCards(levels=4, cube_size=0.1, cube_mass=0.1, ball_mass=0.5):
    """
    Creates a scene with a 1D triangular pyramid stack of cubes (like a house of cards),
    where the bottom has the most cubes, and the top has one,
    stacked vertically and centered, plus a ball positioned next to the base.
    
    Args:
        levels (int): Number of levels (height of the pyramid).
        cube_size (float): Side length of each cube.
        cube_mass (float): Mass of each cube.
        ball_mass (float): Mass of the ball.
    """
    model = pin.Model()
    geom_model = pin.GeometryModel()
    q0_list = []

    cube_shape = hppfcl.Box(cube_size, cube_size, cube_size)
    cube_inertia = pin.Inertia.FromBox(cube_mass, cube_size, cube_size, cube_size)
    ball_radius = cube_size * 1.2
    ball_shape = hppfcl.Sphere(ball_radius)
    ball_inertia = pin.Inertia.FromSphere(ball_mass, ball_radius)

    color_start = np.array([0.2, 0.2, 0.8, 1.0])  # Blue
    color_end = np.array([0.8, 0.2, 0.2, 1.0])    # Red

    def _add_body(name, shape, inertia, placement, color):
        model_item = pin.Model()
        geom_model_item = pin.GeometryModel()
        joint_name = f"joint_{name}"
        jid = model_item.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), joint_name)
        model_item.appendBodyToJoint(jid, inertia, pin.SE3.Identity())
        geom = pin.GeometryObject(name, jid, jid, pin.SE3.Identity(), shape)
        geom.meshColor = color
        geom_model_item.addGeometryObject(geom)

        nonlocal model, geom_model
        model, geom_model = pin.appendModel(
            model, model_item, geom_model, geom_model_item, 0, placement
        )
        q0 = np.concatenate([placement.translation, pin.Quaternion(placement.rotation).coeffs()])
        q0_list.append(q0)

    for level in range(levels):
        num_cubes = levels - level  # Inverted triangle: fewer cubes at higher levels
        z = level * cube_size + cube_size / 2.0

        alpha = level / (levels - 1) if levels > 1 else 1.0
        color = (1 - alpha) * color_start + alpha * color_end

        # Center cubes around x=0 at each level
        x_start = - (num_cubes - 1) * cube_size / 2.0

        for i in range(num_cubes):
            x = x_start + i * cube_size
            pos = np.array([x, 0.0, z])
            placement = pin.SE3(np.eye(3), pos)
            name = f"cube_L{level}_N{i}"
            _add_body(name, cube_shape, cube_inertia, placement, color)

    pyramid_base_half_width = (levels * cube_size) / 2.0
    ball_pos = np.array([
        -(pyramid_base_half_width + ball_radius + cube_size / 2),  # Left of base cubes
        0.0,
        ball_radius  # On the floor
    ])
    ball_placement = pin.SE3(np.eye(3), ball_pos)
    _add_body("ball", ball_shape, ball_inertia, ball_placement, color_start)  # Blue ball

    model.referenceConfigurations["default"] = np.concatenate(q0_list)
    addFloor(geom_model, altitude=0.0)
    geom_model.addAllCollisionPairs()

    return model, geom_model

    
def buildSceneRobotHand(with_item=False, item_size=0.05):
    robot = RobotHand()

    model, geom_model = robot.model, robot.gmodel

    if not with_item:
        return model, geom_model

    # Add a floating capsule
    model_item = pin.Model()
    geom_model_item = pin.GeometryModel()

    shape = hppfcl.Capsule(item_size / 2, item_size)
    world_bounds = np.array([1] * 3 + [np.inf] * 4)

    jid = model_item.addJoint(
        0,
        pin.JointModelFreeFlyer(),
        pin.SE3.Identity(),
        "obj1",
        min_config=-world_bounds,
        max_config=world_bounds,
        max_velocity=np.ones(6),
        max_effort=np.ones(6),
    )
    color = np.random.rand(4)
    color[-1] = 1
    # Place the object so that it is centered
    Mcenter = pin.SE3(np.eye(3), -shape.computeCOM())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geom = pin.GeometryObject(
            f"{str(type(shape))[22:-2]}_{jid}",
            jid,
            jid,
            placement=Mcenter,
            collision_geometry=shape,
        )

    geom.meshColor = np.array(color)
    geom_model_item.addGeometryObject(geom)
    # Add inertia
    volum = shape.computeVolume()
    I3 = shape.computeMomentofInertia()
    density = 700  # kg/m3 = wood
    model_item.appendBodyToJoint(
        jid, pin.Inertia(volum * density, np.zeros(3), I3 * density), pin.SE3.Identity()
    )

    # Merge both model
    model_dual, geom_model_dual = pin.appendModel(
        model, model_item, geom_model, geom_model_item, 0, pin.SE3.Identity()
    )

    # Create the reference configuration for the dual model.
    # The order is suprising, but I guess the cube is attached as a first joint because
    # it is right after the universe joint.
    model_dual.referenceConfigurations["default"] = np.concatenate(
        [
            np.array(
                [0, 0, item_size * 2, -0.36220244, -0.24913755, 0.54623537, 0.71299845]
            ),
            model.referenceConfigurations["default"],
        ]
    )

    return model_dual, geom_model_dual


def buildSceneHouseOfCards(
    levels=2,
    ball_mass=0.1,
    ball_radius=0.03,
    card_mass=0.005,
    seed=0,
    with_ball=True,
):
    """
    Creates a scene with a house of cards and a ball ready to fall on it.

    The house is built procedurally to the specified number of levels, forming a
    pyramid. All cards and the ball are free-flying bodies.

    Args:
        levels (int): The number of levels (floors) in the house of cards.
        ball_mass (float): Mass of the ball.
        ball_radius (float): Radius of the ball.
        card_mass (float): Mass of a single card.
        seed (int): Random seed for reproducibility.
        with_ball (bool): If True, add the falling ball to the scene.

    Returns:
        model, geom_model
    """
    # Set up random seeds
    pin.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Scene and model objects
    model = pin.Model()
    geom_model = pin.GeometryModel()
    q0_list = []

    # --- Parameters ---
    # Card parameters
    CARD_HEIGHT = 0.2
    CARD_WIDTH = 0.12  # This is the 'depth' of the card, along Y
    CARD_THICKNESS = 0.002
    LEAN_ANGLE = np.deg2rad(15.0)  # Angle from the vertical
    CARD_COLOR = np.array([0.9, 0.85, 0.7, 1.0])
    BALL_COLOR = np.array([0.8, 0.2, 0.2, 1.0])

    # Define world boundaries for free-flyer joints
    WORLD_BOUNDS = np.array([2.0] * 3 + [np.inf] * 4)

    # Derived geometry
    z_com_lean = CARD_HEIGHT / 2 * np.cos(LEAN_ANGLE)
    x_com_lean = CARD_HEIGHT / 2 * np.sin(LEAN_ANGLE)
    z_apex = CARD_HEIGHT * np.cos(LEAN_ANGLE)
    x_base_half_width = CARD_HEIGHT * np.sin(LEAN_ANGLE)

    # Layout parameters
    BASE_AFRAME_SEPARATION = (
        x_base_half_width * 2 + 0.01
    )  # Separation between centers of two A-frames
    H_CARD_LENGTH = BASE_AFRAME_SEPARATION

    # --- Shapes and Inertias ---
    card_shape = hppfcl.Box(CARD_THICKNESS, CARD_WIDTH, CARD_HEIGHT)
    card_inertia = pin.Inertia.FromBox(
        card_mass, CARD_THICKNESS, CARD_WIDTH, CARD_HEIGHT
    )
    h_card_shape = hppfcl.Box(H_CARD_LENGTH, CARD_WIDTH, CARD_THICKNESS)
    h_card_mass = card_mass * H_CARD_LENGTH / CARD_HEIGHT
    h_card_inertia = pin.Inertia.FromBox(
        h_card_mass, H_CARD_LENGTH, CARD_WIDTH, CARD_THICKNESS
    )

    # Helper to add a free-flying body to the models
    def _add_body(name, shape, inertia, color):
        jid = model.addJoint(
            0,
            pin.JointModelFreeFlyer(),
            pin.SE3.Identity(),
            f"joint_{name}",
            min_config=-WORLD_BOUNDS,
            max_config=WORLD_BOUNDS,
            max_velocity=np.ones(6),
            max_effort=np.ones(6),
        )
        model.appendBodyToJoint(jid, inertia, pin.SE3.Identity())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geom = pin.GeometryObject(
                name, jid, jid, placement=pin.SE3.Identity(), collision_geometry=shape
            )
        geom.meshColor = color
        geom_model.addGeometryObject(geom)

    # --- Procedurally build the house of cards ---
    z_level_base = 0.0
    M_left_template = pin.SE3(
        pin.rpy.rpyToMatrix(0, LEAN_ANGLE, 0), np.array([-x_com_lean, 0, z_com_lean])
    )
    M_right_template = pin.SE3(
        pin.rpy.rpyToMatrix(0, -LEAN_ANGLE, 0), np.array([x_com_lean, 0, z_com_lean])
    )

    for level in range(levels):
        num_aframes = levels - level
        # Center the row of A-frames at x=0
        x_centers = [
            (i - (num_aframes - 1) / 2.0) * BASE_AFRAME_SEPARATION
            for i in range(num_aframes)
        ]

        # Place the leaning cards (A-frames)
        for i, x_center in enumerate(x_centers):
            for side, M_template in [
                ("L", M_left_template),
                ("R", M_right_template),
            ]:
                name = f"card_L{level}_A{i}_{side}"
                M = M_template.copy()
                M.translation[0] += x_center
                M.translation[2] += z_level_base
                _add_body(name, card_shape, card_inertia, CARD_COLOR)
                q = np.concatenate([M.translation, pin.Quaternion(M.rotation).coeffs()])
                q0_list.append(q)

        if level < levels - 1:
            for i in range(num_aframes - 1):
                name = f"card_H{level}_B{i}"
                M_H = pin.SE3.Identity()
                M_H.translation[0] = (x_centers[i] + x_centers[i + 1]) / 2.0
                M_H.translation[2] = z_level_base + z_apex + CARD_THICKNESS / 2
                _add_body(name, h_card_shape, h_card_inertia, CARD_COLOR)
                q = np.concatenate([M_H.translation, pin.Quaternion(M_H.rotation).coeffs()])
                q0_list.append(q)

        z_level_base += z_apex + CARD_THICKNESS

    if with_ball:
        ball_shape = hppfcl.Sphere(ball_radius)
        ball_inertia = pin.Inertia.FromSphere(ball_mass, ball_radius)
        _add_body("ball", ball_shape, ball_inertia, BALL_COLOR)

        z_top_of_house = (levels - 1) * (z_apex + CARD_THICKNESS) + z_apex
        M_ball = pin.SE3.Identity()
        M_ball.translation[0] = (random.random() - 0.5) * CARD_THICKNESS * 5
        M_ball.translation[2] = z_top_of_house + ball_radius + 0.05
        q_ball = np.concatenate(
            [M_ball.translation, pin.Quaternion(M_ball.rotation).coeffs()]
        )
        q0_list.append(q_ball)

    geom_model.addAllCollisionPairs()
    addFloor(geom_model)

    model.referenceConfigurations["default"] = np.concatenate(q0_list)

    return model, geom_model
def buildSceneHandAndStackedCubes():
    robot_fixed_base = RobotHand()

    # Base models
    model = pin.Model()
    geom_model = pin.GeometryModel()
    visual_model = pin.GeometryModel()

    base_joint_id = model.addJoint(
    0, pin.JointModelFixed(), pin.SE3.Identity(), "hand_base_joint"
    )

    # Create a frame for this joint (BODY type so you can attach things)
    base_frame = pin.Frame(
        "hand_base_frame",
        base_joint_id,
        0,
        pin.SE3.Identity(),
        pin.FrameType.BODY
    )
    base_frame_id = model.addFrame(base_frame)

    model, geom_model = pin.appendModel(
        model,
        robot_fixed_base.model,
        geom_model,
        robot_fixed_base.gmodel,
        base_frame_id,
        pin.SE3.Identity()
    )

    # Append visuals manually to avoid geometry index conflicts
    for visual_obj in robot_fixed_base.visual_model.geometryObjects:
        visual_copy = visual_obj.copy()
        parent_joint_name = robot_fixed_base.model.names[visual_obj.parentJoint]
        parent_joint_id = model.getJointId(parent_joint_name)
        visual_copy.parentJoint = parent_joint_id
        visual_model.addGeometryObject(visual_copy)

    # Cube parameters
    cube_size = 0.04
    cube_mass = 0.1
    cube_shape = hppfcl.Box(cube_size, cube_size, cube_size)
    cube_inertia = pin.Inertia.FromBox(cube_mass, cube_size, cube_size, cube_size)

    cube_stack_height = 0.25 

    cubes_to_add = [
        {"name": "blue_cube", "color": np.array([0.2, 0.2, 0.8, 1.0]), "position": np.array([0.0, 0.0, cube_stack_height])},
        {"name": "red_cube",  "color": np.array([0.8, 0.2, 0.2, 1.0]), "position": np.array([0.0, 0.0, cube_stack_height + cube_size * 1.2])},
    ]

    for cube_info in cubes_to_add:
        model_item = pin.Model()
        geom_item = pin.GeometryModel()

        jid = model_item.addJoint(
            0, pin.JointModelFreeFlyer(), pin.SE3.Identity(),
            f"joint_{cube_info['name']}"
        )
        model_item.appendBodyToJoint(jid, cube_inertia, pin.SE3.Identity())

        geom = pin.GeometryObject(
            cube_info['name'], jid, jid, pin.SE3.Identity(), cube_shape
        )
        geom.meshColor = cube_info['color']
        geom_item.addGeometryObject(geom)

        model, geom_model = pin.appendModel(
            model, model_item, geom_model, geom_item, 0, pin.SE3.Identity()
        )

        visual_geom = geom.copy()
        visual_geom.parentJoint = model.getJointId(f"joint_{cube_info['name']}")
        visual_model.addGeometryObject(visual_geom)

    # Hand's default configuration
    nq_hand_with_base = 7 + robot_fixed_base.model.nq
    q0_hand_with_base = np.zeros(nq_hand_with_base)
    q0_hand_with_base[2] = 0.15  # z-height
    rot_x_neg_90 = pin.rpy.rpyToMatrix(-np.pi / 2, 0, 0)
    q0_hand_with_base[3:7] = pin.Quaternion(rot_x_neg_90).coeffs()
    q0_hand_with_base[7:] = robot_fixed_base.q0

    # Fixed cube poses hardcoded (position + quaternion)
    def pose_to_q(pos):
        return np.concatenate([pos, [0, 0, 0, 1]])  # Identity quaternion

    q0_blue_cube = pose_to_q(cubes_to_add[0]['position'])
    q0_red_cube = pose_to_q(cubes_to_add[1]['position'])

    model.referenceConfigurations["default"] = np.concatenate([
        q0_blue_cube,
        q0_red_cube,
        q0_hand_with_base
    ])

    addFloor(geom_model)

    return model, visual_model



def buildSceneTalosFallingCube():
    robot = load_robot("talos")
    model, geom_model = robot.model, robot.collision_model

    cube_size = 0.4
    cube_mass = 5.0
    cube_shape = hppfcl.Box(cube_size, cube_size, cube_size)
    cube_inertia = pin.Inertia.FromBox(cube_mass, cube_size, cube_size, cube_size)

    model_item = pin.Model()
    geom_model_item = pin.GeometryModel()
    jid = model_item.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "joint_cube")
    model_item.appendBodyToJoint(jid, cube_inertia, pin.SE3.Identity())
    geom = pin.GeometryObject("red_cube", jid, jid, pin.SE3.Identity(), cube_shape)
    geom.meshColor = np.array([0.8, 0.2, 0.2, 1.0])
    geom_model_item.addGeometryObject(geom)
    
    model, geom_model = pin.appendModel(
        model, model_item, geom_model, geom_model_item, 0, pin.SE3.Identity()
    )

    q0_robot = robot.q0.copy()

    cube_pos = np.array([-0.2, 0.0, 2.0]) 
    
    cube_quat = pin.Quaternion.Identity().coeffs()
    q0_cube = np.concatenate([cube_pos, cube_quat])

    model.referenceConfigurations["default"] = np.concatenate([q0_robot, q0_cube])

    geom_model.addAllCollisionPairs()
    addFloor(geom_model, altitude=0.0)

    return model, geom_model

def buildSceneQuadrupedOnHills():
    """
    Builds the scene: a quadruped robot on steep terrain.

    This version uses a single, large, tilted box for the terrain to be
    efficient and prevent memory crashes.
    """
    robot = load_robot("go2")
    model, geom_model = robot.model, robot.collision_model

    robot_color = np.array([0.8, 0.8, 0.2, 1.0]) 

    for g_obj in geom_model.geometryObjects:
        if "go2" in g_obj.name:
            g_obj.meshColor = robot_color

    terrain_size = [5.0, 5.0, 0.1] 
    terrain_shape = hppfcl.Box(*terrain_size)


    terrain_angle = np.deg2rad(20.0) 
    terrain_placement = pin.SE3.Identity()
    terrain_placement.rotation = pin.rpy.rpyToMatrix(0, -terrain_angle, 0)

    terrain_geom = pin.GeometryObject("steep_terrain", 0, 0, terrain_placement, terrain_shape)
    terrain_geom.meshColor = np.array([0.1, 0.1, 0.1, 0.8])
    terrain_id = geom_model.addGeometryObject(terrain_geom)

    for g_obj in geom_model.geometryObjects:
        if "terrain" not in g_obj.name:
             geom_model.addCollisionPair(pin.CollisionPair(geom_model.getGeometryId(g_obj.name), terrain_id))

    q0 = robot.q0.copy()
    q0[0] = 1.0
    q0[2] = 1.0
    base_rotation = pin.rpy.rpyToMatrix(0, -terrain_angle, 0)
    q0[3:7] = pin.Quaternion(base_rotation).coeffs()
    
    model.referenceConfigurations["default"] = q0
    
    return model, geom_model

def buildSceneTriangleAndBall(levels=4, cube_size=0.2, cube_mass=0.1, ball_mass=0.5):
    model = pin.Model()
    geom_model = pin.GeometryModel()
    q0_list = []

    cube_shape = hppfcl.Box(cube_size, cube_size, cube_size)
    cube_inertia = pin.Inertia.FromBox(cube_mass, cube_size, cube_size, cube_size)
    ball_radius = cube_size * 1.2
    ball_shape = hppfcl.Sphere(ball_radius)
    ball_inertia = pin.Inertia.FromSphere(ball_mass, ball_radius)

    color_start = np.array([0.2, 0.2, 0.8, 1.0])  # Blue
    color_end = np.array([0.8, 0.2, 0.2, 1.0])    # Red

    def _add_body(name, shape, inertia, placement, color):
        model_item = pin.Model()
        geom_model_item = pin.GeometryModel()
        joint_name = f"joint_{name}"
        jid = model_item.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), joint_name)
        model_item.appendBodyToJoint(jid, inertia, pin.SE3.Identity())
        geom = pin.GeometryObject(name, jid, jid, pin.SE3.Identity(), shape)
        geom.meshColor = color
        geom_model_item.addGeometryObject(geom)

        nonlocal model, geom_model
        model, geom_model = pin.appendModel(
            model, model_item, geom_model, geom_model_item, 0, placement
        )
        q0 = np.concatenate([placement.translation, pin.Quaternion(placement.rotation).coeffs()])
        q0_list.append(q0)

    overlap = 0.001 

    for level in range(levels):
        num_cubes = levels - level  
        z = -0.15 + level * cube_size/2 + cube_size 

        alpha = level / (levels - 1) if levels > 1 else 1.0
        color = (1 - alpha) * color_start + alpha * color_end
        
        x_start = - (num_cubes - 1) * (cube_size - overlap) / 2.0

        for i in range(num_cubes):
            x = x_start + i * (cube_size - overlap) 
            
            pos = np.array([x, 0.0, z])
            placement = pin.SE3(np.eye(3), pos)
            name = f"cube_L{level}_N{i}"
            _add_body(name, cube_shape, cube_inertia, placement, color)

    ball_pos = np.array([
        0.0,                                     
        (cube_size / 2) + ball_radius + 0.05,    
        ball_radius                              
    ])
    ball_placement = pin.SE3(np.eye(3), ball_pos)
    _add_body("ball", ball_shape, ball_inertia, ball_placement, color_start)

    model.referenceConfigurations["default"] = np.concatenate(q0_list)
    addFloor(geom_model, altitude=0.0)
    geom_model.addAllCollisionPairs()

    return model, geom_model

class MyTest(unittest.TestCase):
    def test_pillsbox(self):
        model, gmodel = buildScenePillsBox(nobj=10)
        assert isinstance(model, pin.Model)

    def test_3b(self):
        model, gmodel = buildSceneThreeBodies()
        assert isinstance(model, pin.Model)

    def test_cubes(self):
        model, gmodel = buildSceneCubes(3)
        assert isinstance(model, pin.Model)
        # assert model.nv == 3 * 6
        # assert len(gmodel.geometryObjects) == 3 * 9
        # assert len(gmodel.collisionPairs) == 3 * 8**2

    def test_floor(self):
        model, gmodel = buildSceneThreeBodies()
        addFloor(gmodel, True)
        assert isinstance(gmodel, pin.GeometryModel)
        assert len(gmodel.geometryObjects) == 4
        assert len(gmodel.collisionPairs) == 6

    def test_robot_hand(self):
        model, gmodel = buildSceneRobotHand()
        assert isinstance(gmodel, pin.GeometryModel)
        assert len(gmodel.geometryObjects) == 18
        assert len(gmodel.collisionPairs) == 45

    def test_robot_hand_plus(self):
        model, gmodel = buildSceneRobotHand(True)
        assert isinstance(gmodel, pin.GeometryModel)
        assert len(gmodel.geometryObjects) == 19
        assert len(gmodel.collisionPairs) == 63

    def test_house_of_cards(self):
        levels = 2
        model, gmodel = buildSceneHouseOfCards(levels=levels, with_ball=True)
        assert isinstance(model, pin.Model)

        num_leaning_cards = levels * (levels + 1)  # 2 cards per A-frame
        num_horizontal_cards = (levels - 1) * levels / 2
        num_cards = int(num_leaning_cards + num_horizontal_cards)
        num_bodies = num_cards + 1  # +1 for the ball

        self.assertEqual(model.njoints, num_bodies + 1)  # +1 for universe
        self.assertEqual(model.nv, num_bodies * 6)
        # num_bodies + 1 floor
        self.assertEqual(len(gmodel.geometryObjects), num_bodies + 1)
        # nC2(num_bodies) pairs between bodies + num_bodies pairs with floor
        expected_pairs = int(num_bodies * (num_bodies - 1) / 2 + num_bodies)
        self.assertEqual(len(gmodel.collisionPairs), expected_pairs)

