"""
This file implements 2 helpers to simply visualize a set of contact models in Pinocchio.
For each contact model, the vizualisation is composed of a pair of (red) disks
placed parallel to the contact surfaces of the two collision objects.
The contact models are typically created from the collision/distance results of
hppfcl, e.g. using create_rigid_contact_models_for_hppfcl.

The file implements 2 main methods:

- preallocateVisualObjects: add visual objects to render the collision, as
small red disks "collision patches" on both side of the collision (on each
objects of the collision pairs). These objects are pre-defined in advance, and
you must specify in advance the max number of colpatch
- updateVisualObjects: move the collision patches in the visual model. If the
visualizer is provided, also automatically hide the unecessary patches.

"""

import hppfcl
import numpy as np
import pinocchio as pin

# ### HYPERPARAMETERS OF THE DISPLAY HELPERS
COLPATCH_RADIUS = 0.05
COLPATCH_THICKNESS = 3e-3
COLPATCH_COLOR = np.array([1, 0, 0, 0.5])
COLPATCH_TEMPLATE_NAME = "collpatch_{ncolpatch}_{first_or_second}"
COLPATCH_DEFAULT_PREALLOC = 10

# Create once for all the pin.GeometryObject used as colpatch visual.
shape = hppfcl.Cylinder(COLPATCH_RADIUS, COLPATCH_THICKNESS)
geom = pin.GeometryObject(
    "waitforit", 0, 0, placement=pin.SE3.Identity(), collision_geometry=shape
)
geom.meshColor = COLPATCH_COLOR


def _createVisualObjects(visual_model, ncolpatch, verbose=False):
    """
    Create a single pair of patches. cammed by preallocateVisualObjects().
    """
    if verbose:
        print(f"   --- Create a new visual patch #{ncolpatch}")
    geom.name = COLPATCH_TEMPLATE_NAME.format(
        ncolpatch=ncolpatch, first_or_second="first"
    )
    firstId = visual_model.addGeometryObject(geom)
    geom.name = COLPATCH_TEMPLATE_NAME.format(
        ncolpatch=ncolpatch, first_or_second="second"
    )
    secondId = visual_model.addGeometryObject(geom)
    return firstId, secondId


def _whatIsMyVisualizer(visualizer):
    """
    Return a string defining the type of visualizer (none, meshcat, possibly
    gepettoviewer in the future).

    """
    if visualizer is None:
        return "none"
    try:
        import meshcat

        if isinstance(visualizer.viewer, meshcat.visualizer.Visualizer):
            return "meshcat"
    except ModuleNotFoundError:
        pass
    except AttributeError:
        pass


def _meshcat_hideColPatches(
    visual_model, visualizer, ntokeep, objectsAreSorted=True, verbose=False
):
    """
    There is no generic way to hide/show visuals for all viewers of Pinocchio.
    This is the specific instance for the meshcat viewer.  If objectsAreSorted
    is true, then assume that the visual objects are sorted and in final
    position in the visual_model.geometryObjects list and make the
    seek-and-hide faster.
    """
    assert "viewer" in dir(visualizer)
    colpatchKey = COLPATCH_TEMPLATE_NAME.split("_")[0]

    if objectsAreSorted:
        name0 = COLPATCH_TEMPLATE_NAME.format(ncolpatch=0, first_or_second="first")
        idx0 = visual_model.getGeometryId(name0)
        for ig, g in enumerate(visual_model.geometryObjects[idx0:]):
            vis = (ig // 2) <= ntokeep
            adr = visualizer.getViewerNodeName(g, pin.VISUAL)
            visualizer.viewer[adr].set_property("visible", vis)
            if verbose:
                print(f"Make {ig//2} ({adr}) visible : {vis}")
    else:
        for g in visual_model.geometryObjects:
            if colpatchKey in g.name:
                ref = g.name.split("_")[1]
                try:
                    ref = int(ref)
                except ValueError:
                    print(f"Unexpected error with the colpatch name {g.name} ")
                    print(
                        "It was expected to follow the template "
                        + f"{COLPATCH_TEMPLATE_NAME}"
                    )
                    continue

                vis = ref <= ntokeep
                adr = visualizer.getViewerNodeName(g, pin.VISUAL)
                visualizer.viewer[adr].set_property("visible", vis)
                if verbose:
                    print(f"Make {ref} ({adr}) visible : {vis}")


def preallocateVisualObjects(
    visual_model, number=COLPATCH_DEFAULT_PREALLOC, verbose=False
):
    """
    Create the visual objects.
    This must be called before calling updateVisualObjects().
    """
    for ic in range(number):
        _createVisualObjects(visual_model, ic, verbose=verbose)


def updateVisualObjects(
    model, data, contact_models, contact_datas, visual_model, visualizer=None
):
    """
    Take the contact models list(pin.RigidConstraintModels) and update the
    placement of the visual objects in visual_model.  In addition, it can hide
    the objects that are not useful, but this action is specific to the viewer
    and needs you to also pass the viewer (only for meshcat for now, but
    implementing it for Gepetto-viewer would be easy).

    """
    ic = -1
    for ic, [cmodel, cdata] in enumerate(zip(contact_models, contact_datas)):
        name_first = COLPATCH_TEMPLATE_NAME.format(
            ncolpatch=ic, first_or_second="first"
        )
        gid_first = visual_model.getGeometryId(name_first)
        name_second = COLPATCH_TEMPLATE_NAME.format(
            ncolpatch=ic, first_or_second="second"
        )
        gid_second = visual_model.getGeometryId(name_second)

        if gid_first == len(visual_model.geometryObjects):
            print(
                "There is not enough pre-loaded colpatch for displaying all collisions!"
            )
            break

        assert gid_first < visual_model.ngeoms
        assert "collpatch" in visual_model.geometryObjects[gid_first].name
        assert gid_second < visual_model.ngeoms
        assert "collpatch" in visual_model.geometryObjects[gid_second].name

        cdata.oMc1 = data.oMi[cmodel.joint1_id] * cmodel.joint1_placement
        cdata.oMc2 = data.oMi[cmodel.joint2_id] * cmodel.joint2_placement
        visual_model.geometryObjects[gid_first].placement = cdata.oMc1
        visual_model.geometryObjects[gid_second].placement = cdata.oMc2

    vizType = _whatIsMyVisualizer(visualizer)
    if vizType == "meshcat":
        _meshcat_hideColPatches(visual_model, visualizer, ic)


# ### TESTING ZONE
# ### TESTING ZONE
# ### TESTING ZONE
