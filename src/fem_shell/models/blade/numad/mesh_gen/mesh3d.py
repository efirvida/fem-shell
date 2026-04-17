import numpy as np
from scipy import interpolate

from fem_shell.models.blade.numad.mesh_gen.spatial_grid_list3d import *


class Mesh3D:
    def __init__(self, boundaryNodes, boundaryFaces=[]):
        self.nodeGL = None
        self.faceGL = None
        self.tetElGL = None

        self.minFaceArea = 0.0
        self.maxFaceArea = 1.0
        self.avgProjLen = 1.0

        self.numBndNodes = len(boundaryNodes)
        self.numNodes = self.numBndNodes
        self.ndSize = self.numBndNodes
        self.nodes = np.array(boundaryNodes)

        self.numBndFaces = len(boundaryFaces)
        self.faceNodes = np.array(boundaryFaces)
        self.numFaces = self.numBndFaces
        self.faceSize = self.numBndFaces
        self.faceElements = np.array([])
        self.faceUnitNorms = np.array([])

        self.numTetEls = 0
        self.tetElSize = 0
        self.tetElements = np.array([])

        self.numWedgeEls = 0
        self.wedgeElSize = 0
        self.wedgeElements = np.array([])

        self.numHexEls = 0
        self.hexElSize = 0
        self.hexElements = np.array([])

    def createSweptMesh(
        self,
        sweepMethod,
        sweepElements,
        sweepDistance=1.0,
        point=[],
        axis=[],
        followNormal=False,
        destNodes=[],
        interpMethod="linear",
    ):
        ## sweepMethod = inDirection, toPoint, fromPoint, toDestNodes, revolve
        """Object data modified: self.quadElements, self.nodes, self.quadElements
        Parameters
        ----------

        Returns
        -------
        nodes
        elements
        """
        nbNds = self.numBndNodes
        try:
            totSweepEls = sum(sweepElements)
            ndSize = nbNds * (totSweepEls + 1)
            stages = len(sweepElements)
            multiStage = True
        except:
            totSweepEls = sweepElements
            ndSize = nbNds * (sweepElements + 1)
            stages = 1
            multiStage = False
        tmp = self.nodes.copy()
        self.nodes = np.zeros((ndSize, 3))
        self.nodes[0:nbNds] = tmp
        self.ndSize = ndSize
        self.numNodes = nbNds

        nbFcs = self.numBndFaces
        elSize = nbFcs * totSweepEls

        self.wedgeElements = -np.ones((elSize, 6), dtype=int)
        self.wedgeElSize = elSize
        self.numWedgeEls = 0
        wERank = np.zeros(elSize, dtype=int)

        self.hexElements = -np.ones((elSize, 8), dtype=int)
        self.hexElSize = elSize
        self.numHexEls = 0
        hERank = np.zeros(elSize, dtype=int)

        methString = "inDirection toPoint fromPoint"
        if sweepMethod in methString:
            ndDir = list()
            if sweepMethod == "inDirection":
                mag = np.linalg.norm(axis)
                unitAxis = (1.0 / mag) * np.array(axis)
                for i in range(0, self.numNodes):
                    ndDir.append(unitAxis)
            else:
                pAr = np.array(point)
                for i in range(0, self.numNodes):
                    if sweepMethod == "toPoint":
                        vec = pAr - nd
                    else:
                        vec = nd - pAr
                    mag = np.linalg.norm(vec)
                    unitVec = (1.0 / mag) * vec
                    ndDir.append(unitVec)
            rowNds = self.numNodes
            rowEls = self.numFaces
            stepLen = sweepDistance / sweepElements
            nNds = self.numNodes
            wE = self.numWedgeEls
            hE = self.numHexEls
            eli = 0
            for i in range(0, sweepElements):
                for j in range(0, rowNds):
                    newNd = self.nodes[j] + (i + 1) * stepLen * ndDir[j]
                    self.nodes[nNds] = newNd
                    nNds = nNds + 1
                for j in range(0, rowEls):
                    n1 = self.faceNodes[j, 0] + i * rowNds
                    n2 = self.faceNodes[j, 1] + i * rowNds
                    n3 = self.faceNodes[j, 2] + i * rowNds
                    if self.faceNodes[j, 3] == -1:
                        n4 = n1 + rowNds
                        n5 = n2 + rowNds
                        n6 = n3 + rowNds
                        self.wedgeElements[wE] = np.array([n1, n2, n3, n4, n5, n6])
                        wERank[wE] = eli
                        wE = wE + 1
                    else:
                        n4 = self.faceNodes[j, 3] + i * rowNds
                        n5 = n1 + rowNds
                        n6 = n2 + rowNds
                        n7 = n3 + rowNds
                        n8 = n4 + rowNds
                        self.hexElements[hE] = np.array([n1, n2, n3, n4, n5, n6, n7, n8])
                        hERank[hE] = eli
                        hE = hE + 1
                    eli = eli + 1
            self.numNodes = nNds
            self.numWedgeEls = wE
            self.numHexEls = hE
        elif sweepMethod == "toDestNodes":
            nNds = self.numNodes
            nbNds = self.numBndNodes
            wE = self.numWedgeEls
            hE = self.numHexEls
            eli = 0
            if not multiStage:
                sweepElements = [sweepElements]
                destNodes = [destNodes]
            if interpMethod == "linear":
                prevDest = self.nodes.copy()
                cumElLay = 0
                for stg in range(0, stages):
                    dNds = np.array(destNodes[stg])
                    ndDir = np.zeros((nbNds, 3))
                    for ndi in range(0, nbNds):
                        ndDir[ndi] = (1.0 / sweepElements[stg]) * (dNds[ndi] - prevDest[ndi])
                    for i in range(0, sweepElements[stg]):
                        for ndi in range(0, nbNds):
                            self.nodes[nNds] = self.nodes[nNds - nbNds] + ndDir[ndi]
                            nNds = nNds + 1
                        for fci in range(0, self.numBndFaces):
                            n1 = self.faceNodes[fci, 0] + cumElLay * nbNds
                            n2 = self.faceNodes[fci, 1] + cumElLay * nbNds
                            n3 = self.faceNodes[fci, 2] + cumElLay * nbNds
                            if self.faceNodes[fci, 3] == -1:
                                n4 = n1 + nbNds
                                n5 = n2 + nbNds
                                n6 = n3 + nbNds
                                self.wedgeElements[wE] = np.array([n1, n2, n3, n4, n5, n6])
                                wERank[wE] = eli
                                wE = wE + 1
                            else:
                                n4 = self.faceNodes[fci, 3] + cumElLay * nbNds
                                n5 = n1 + nbNds
                                n6 = n2 + nbNds
                                n7 = n3 + nbNds
                                n8 = n4 + nbNds
                                self.hexElements[hE] = np.array([n1, n2, n3, n4, n5, n6, n7, n8])
                                hERank[hE] = eli
                                hE = hE + 1
                            eli = eli + 1
                        cumElLay = cumElLay + 1
                    prevDest = dNds.copy()
            else:  ## Smooth interpolation
                xMat = np.zeros((nbNds, totSweepEls + 1))
                yMat = np.zeros((nbNds, totSweepEls + 1))
                zMat = np.zeros((nbNds, totSweepEls + 1))
                pDest = (1.0 / stages) * np.array(range(0, stages + 1))
                pAll = (1.0 / totSweepEls) * np.array(range(0, totSweepEls + 1))
                for ndi in range(0, nbNds):
                    xDest = [self.nodes[ndi, 0]]
                    yDest = [self.nodes[ndi, 1]]
                    zDest = [self.nodes[ndi, 2]]
                    for dNds in destNodes:
                        xDest.append(dNds[ndi][0])
                        yDest.append(dNds[ndi][1])
                        zDest.append(dNds[ndi][2])
                    xDest = np.array(xDest)
                    iFun = interpolate.interp1d(
                        pDest,
                        xDest,
                        "cubic",
                        axis=0,
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    xAll = iFun(pAll)
                    xMat[ndi, :] = xAll
                    yDest = np.array(yDest)
                    iFun = interpolate.interp1d(
                        pDest,
                        yDest,
                        "cubic",
                        axis=0,
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    yAll = iFun(pAll)
                    yMat[ndi, :] = yAll
                    zDest = np.array(zDest)
                    iFun = interpolate.interp1d(
                        pDest,
                        zDest,
                        "cubic",
                        axis=0,
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    zAll = iFun(pAll)
                    zMat[ndi, :] = zAll
                eli = 0
                for i in range(0, totSweepEls):
                    for ndi in range(0, nbNds):
                        self.nodes[nNds] = np.array(
                            [xMat[ndi, i + 1], yMat[ndi, i + 1], zMat[ndi, i + 1]]
                        )
                        nNds = nNds + 1
                    for fci in range(0, self.numBndFaces):
                        n1 = self.faceNodes[fci, 0] + i * nbNds
                        n2 = self.faceNodes[fci, 1] + i * nbNds
                        n3 = self.faceNodes[fci, 2] + i * nbNds
                        if self.faceNodes[fci, 3] == -1:
                            n4 = n1 + nbNds
                            n5 = n2 + nbNds
                            n6 = n3 + nbNds
                            self.wedgeElements[wE] = np.array([n1, n2, n3, n4, n5, n6])
                            wERank[wE] = eli
                            wE = wE + 1
                        else:
                            n4 = self.faceNodes[fci, 3] + i * nbNds
                            n5 = n1 + nbNds
                            n6 = n2 + nbNds
                            n7 = n3 + nbNds
                            n8 = n4 + nbNds
                            self.hexElements[hE] = np.array([n1, n2, n3, n4, n5, n6, n7, n8])
                            hERank[hE] = eli
                            hE = hE + 1
                        eli = eli + 1
            self.numNodes = nNds
            self.numWedgeEls = wE
            self.numHexEls = hE

        for eli in range(0, self.numWedgeEls):
            n1 = self.wedgeElements[eli, 0]
            n2 = self.wedgeElements[eli, 1]
            n3 = self.wedgeElements[eli, 2]
            n4 = self.wedgeElements[eli, 3]
            v1 = self.nodes[n2] - self.nodes[n1]
            v2 = self.nodes[n3] - self.nodes[n1]
            v3 = self.nodes[n4] - self.nodes[n1]
            mat = np.array([v1, v2, v3])
            detM = np.linalg.det(mat)
            if detM < 0.0:
                self.wedgeElements[eli, 1] = n3
                self.wedgeElements[eli, 2] = n2
                sw = self.wedgeElements[eli, 4]
                self.wedgeElements[eli, 4] = self.wedgeElements[eli, 5]
                self.wedgeElements[eli, 5] = sw

        for eli in range(0, self.numHexEls):
            n1 = self.hexElements[eli, 0]
            n2 = self.hexElements[eli, 1]
            n4 = self.hexElements[eli, 3]
            n5 = self.hexElements[eli, 4]
            v1 = self.nodes[n2] - self.nodes[n1]
            v2 = self.nodes[n4] - self.nodes[n1]
            v3 = self.nodes[n5] - self.nodes[n1]
            mat = np.array([v1, v2, v3])
            detM = np.linalg.det(mat)
            if detM < 0.0:
                self.hexElements[eli, 1] = n4
                self.hexElements[eli, 3] = n2
                sw = self.hexElements[eli, 5]
                self.hexElements[eli, 5] = self.hexElements[eli, 7]
                self.hexElements[eli, 7] = sw

        meshOut = dict()
        meshOut["nodes"] = self.nodes
        totEls = self.numWedgeEls + self.numHexEls
        allEls = -np.ones((totEls, 8), dtype=int)
        for eli in range(0, self.numWedgeEls):
            allEls[wERank[eli], 0:6] = self.wedgeElements[eli]
        for eli in range(0, self.numHexEls):
            allEls[hERank[eli], 0:8] = self.hexElements[eli]
        # if(self.numWedgeEls > 0):
        # allEls[0:self.numWedgeEls,0:6] = self.wedgeElements[0:self.numWedgeEls]
        # if(self.numHexEls > 0):
        # allEls[self.numWedgeEls:totEls,0:8] = self.hexElements[0:self.numHexEls]
        meshOut["elements"] = allEls
        return meshOut


def create_offset_layers(nodes, elements, n_layers, first_thickness, growth_rate):
    """Extrude a quad4 surface mesh outward to build a hex8 boundary-layer mesh.

    Each node is moved outward along its averaged vertex normal by a
    distance that grows geometrically between layers.  Layer *k* (0-based)
    has thickness ``first_thickness * growth_rate**k``.

    Parameters
    ----------
    nodes : ndarray, shape (N, 3)
        Surface node coordinates [x, y, z].
    elements : ndarray, shape (M, 4)
        Quad4 connectivity, 0-based indices.  Elements with ``el[3] == -1``
        (degenerate triangles) are skipped and excluded from the output.
    n_layers : int
        Number of hex layers to extrude.
    first_thickness : float
        Thickness of the first (innermost) layer.
    growth_rate : float
        Geometric growth factor between consecutive layers.

    Returns
    -------
    dict with keys:
        ``nodes``    — ndarray shape (N*(n_layers+1), 3)
        ``elements`` — ndarray shape (M_quad * n_layers, 8), hex8 connectivity
        ``sets``     — dict with node sets:
                        ``bladeWallNodes``     (surface node indices, 0..N-1)
                        ``outerBoundaryNodes`` (outermost layer indices)
    """
    from fem_shell.models.blade.numad.mesh_gen.element_utils import get_vertex_normals

    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    N = len(nodes)

    # Only pure quad4 elements participate in the extrusion
    quad_mask = elements[:, 3] != -1
    quad_els = elements[quad_mask]
    M = len(quad_els)

    normals = get_vertex_normals(nodes, elements)  # (N, 3) outward unit normals

    # Build all node layers: layer 0 = original surface
    # Cumulative offset for layer k+1 = sum_{i=0}^{k} first_thickness * growth_rate^i
    all_layers = [nodes.copy()]
    current = nodes.copy()
    for k in range(n_layers):
        thickness = first_thickness * (growth_rate ** k)
        current = current + normals * thickness
        all_layers.append(current.copy())

    all_nodes = np.vstack(all_layers)  # shape (N*(n_layers+1), 3)

    # Hex8 connectivity: for each quad and each layer k -> k+1
    # Hex8 node order (VTK/OpenFOAM convention):
    #   bottom face (layer k):   n0, n1, n2, n3
    #   top face   (layer k+1): n0', n1', n2', n3'
    hex_els = np.empty((M * n_layers, 8), dtype=int)
    for k in range(n_layers):
        base_bot = k * N
        base_top = (k + 1) * N
        offset = k * M
        hex_els[offset : offset + M, 0] = quad_els[:, 0] + base_bot
        hex_els[offset : offset + M, 1] = quad_els[:, 1] + base_bot
        hex_els[offset : offset + M, 2] = quad_els[:, 2] + base_bot
        hex_els[offset : offset + M, 3] = quad_els[:, 3] + base_bot
        hex_els[offset : offset + M, 4] = quad_els[:, 0] + base_top
        hex_els[offset : offset + M, 5] = quad_els[:, 1] + base_top
        hex_els[offset : offset + M, 6] = quad_els[:, 2] + base_top
        hex_els[offset : offset + M, 7] = quad_els[:, 3] + base_top

    # Jacobian sign check: ensure det > 0 for each hex (swap n1/n3 if needed)
    for ei in range(len(hex_els)):
        n0, n1, n3, n4 = hex_els[ei, 0], hex_els[ei, 1], hex_els[ei, 3], hex_els[ei, 4]
        v1 = all_nodes[n1] - all_nodes[n0]
        v2 = all_nodes[n3] - all_nodes[n0]
        v3 = all_nodes[n4] - all_nodes[n0]
        if np.linalg.det(np.array([v1, v2, v3])) < 0:
            hex_els[ei, 1], hex_els[ei, 3] = hex_els[ei, 3], hex_els[ei, 1]
            hex_els[ei, 5], hex_els[ei, 7] = hex_els[ei, 7], hex_els[ei, 5]

    wall_indices = list(range(N))
    outer_indices = list(range(n_layers * N, (n_layers + 1) * N))

    return {
        "nodes": all_nodes,
        "elements": hex_els,
        "sets": {
            "node": [
                {"name": "bladeWallNodes", "labels": wall_indices},
                {"name": "outerBoundaryNodes", "labels": outer_indices},
            ]
        },
    }



def create_outer_domain_unstructured(
    outer_nodes,
    oml_elements,
    cylinder_radius,
    element_size,
    n_cyl_theta=60,
    outer_size_factor=3.0,
    node_z_ref=None,
):
    """Mesh the volume between the BL outer surface and a coaxial cylinder.

    Uses per-slab 3D Delaunay triangulation (scipy).  No gmsh required.

    The inner boundary of the outer domain shares nodes EXACTLY with
    *outer_nodes* (local indices 0..N_inner-1), so the combined mesh has
    a single node array with no duplication and no interface.

    Parameters
    ----------
    outer_nodes : ndarray, shape (N_inner, 3)
        Positions of the BL outermost layer (``outerBoundaryNodes``).
    oml_elements : ndarray, shape (M, 4)
        Quad4 OML connectivity, 0-based indices into *outer_nodes*.
    cylinder_radius : float
        Radius of the outer cylinder from the local section centroid [m].
    element_size : float
        Reference element size (used only for reporting; the mesh density
        is determined by the input node density and ``n_cyl_theta``).
    n_cyl_theta : int
        Angular resolution of the cylinder discretisation (default 60).
    outer_size_factor : float
        Unused in this implementation (kept for API compatibility).
    node_z_ref : ndarray of float, shape (N_inner,), optional
        Reference Z coordinates used **only** for spanwise grouping.
        Must be supplied when the BL extrusion has a non-negligible
        Z component (twisted/pre-bent blades), otherwise the extruded
        Z values scatter nodes from the same spanwise station across
        many thin Z-groups and the inner-profile polygon degenerates.
        Pass ``bl["nodes"][:N_oml, 2]`` (the un-extruded blade-surface Z).

    Returns
    -------
    dict with keys
        ``nodes``    — (N_total, 3) with outer_nodes at indices 0..N_inner-1.
        ``elements`` — (M_tet, 4) int32 tet4 connectivity, local indices.
        ``sets``     — node sets: ``cylinderSurface``.
    """
    import math
    from scipy.spatial import Delaunay
    from shapely.geometry import Polygon as _SPoly, Point as _SPt

    outer_nodes = np.asarray(outer_nodes, dtype=float)
    N_inner = len(outer_nodes)

    # ------------------------------------------------------------------ #
    # Z levels                                                             #
    # Use node_z_ref (un-extruded blade-surface Z) for grouping when      #
    # available.  After BL extrusion with a twisted blade the outward      #
    # normals have a Z component, so the extruded outer_nodes[:, 2] spread #
    # nodes from the same spanwise station across many Z values, making    #
    # each per-Z group have only 1-3 nodes and a degenerate polygon.       #
    # ------------------------------------------------------------------ #
    z_group = np.asarray(node_z_ref, dtype=float) if node_z_ref is not None \
              else outer_nodes[:, 2]

    z = outer_nodes[:, 2]          # actual Z (used for cylinder placement)
    z_span = max(z_group.max() - z_group.min(), 1e-12)
    tol_z = max(1e-6 * z_span, 1e-12)
    z_int = np.round(z_group / tol_z).astype(np.int64)
    unique_zi = np.unique(z_int)
    N_z = len(unique_zi)

    sec_cx = np.empty(N_z); sec_cy = np.empty(N_z); z_vals = np.empty(N_z)
    for k, zi in enumerate(unique_zi):
        mask = z_int == zi
        sec_cx[k] = outer_nodes[mask, 0].mean()
        sec_cy[k] = outer_nodes[mask, 1].mean()
        z_vals[k]  = outer_nodes[mask, 2].mean()

    # Validate cylinder radius
    for k, zi in enumerate(unique_zi):
        mask = z_int == zi
        r_max = np.hypot(outer_nodes[mask, 0] - sec_cx[k],
                         outer_nodes[mask, 1] - sec_cy[k]).max()
        if r_max >= cylinder_radius:
            raise ValueError(
                f"create_outer_domain_unstructured: at Z={z_vals[k]:.3f} m "
                f"some nodes are at r={r_max:.3f} >= cylinder_radius={cylinder_radius}. "
                "Increase cylinder_radius."
            )

    # ------------------------------------------------------------------ #
    # Cylinder nodes                                                       #
    # CYL node (k, j) -> global index CYL0 + k*n_cyl_theta + j           #
    # ------------------------------------------------------------------ #
    theta = np.linspace(0, 2.0 * math.pi, n_cyl_theta, endpoint=False)
    CYL0 = N_inner
    cyl_xyz = np.array([
        [sec_cx[k] + cylinder_radius * math.cos(th),
         sec_cy[k] + cylinder_radius * math.sin(th),
         z_vals[k]]
        for k in range(N_z)
        for th in theta
    ])
    all_nodes_out = np.vstack([outer_nodes, cyl_xyz])

    # ------------------------------------------------------------------ #
    # Per-level inner profile data                                         #
    # ------------------------------------------------------------------ #
    inner_idx_per_z = [np.where(z_int == zi)[0] for zi in unique_zi]

    def _make_poly(pts_2d):
        from shapely.geometry import MultiPoint as _SMPt
        hull = _SMPt(pts_2d.tolist()).convex_hull
        if hull.geom_type == "Polygon":
            return hull
        # Degenerate section (tip/root with < 3 non-collinear pts): tiny buffer
        return hull.buffer(1e-6)

    inner_poly_per_z = [_make_poly(outer_nodes[idx, :2])
                        for idx in inner_idx_per_z]

    # ------------------------------------------------------------------ #
    # Per-slab 3D Delaunay → tet4                                         #
    # For each slab k -> k+1:                                             #
    #   points = inner_k ∪ inner_{k+1} ∪ cyl_k ∪ cyl_{k+1}             #
    #   filter: centroid in Z-slab AND outside inner profile              #
    # ------------------------------------------------------------------ #
    all_tets = []

    for k in range(N_z - 1):
        i_k  = inner_idx_per_z[k]           # global indices in all_nodes_out
        i_k1 = inner_idx_per_z[k + 1]
        c_k  = np.arange(CYL0 + k * n_cyl_theta,
                         CYL0 + (k + 1) * n_cyl_theta)
        c_k1 = np.arange(CYL0 + (k + 1) * n_cyl_theta,
                         CYL0 + (k + 2) * n_cyl_theta)

        slab_global = np.concatenate([i_k, i_k1, c_k, c_k1])
        slab_pts    = all_nodes_out[slab_global]   # (N_slab, 3)

        try:
            dt = Delaunay(slab_pts)
        except Exception:
            continue

        z_k  = z_vals[k]
        z_k1 = z_vals[k + 1]
        poly_k  = inner_poly_per_z[k]
        poly_k1 = inner_poly_per_z[k + 1]

        for simplex in dt.simplices:
            c = slab_pts[simplex].mean(0)

            # Reject tets whose centroid is outside the Z slab
            if c[2] < z_k - tol_z or c[2] > z_k1 + tol_z:
                continue

            # Reject tets whose centroid is inside the blade profile
            t_frac = (c[2] - z_k) / max(z_k1 - z_k, 1e-30)
            poly_check = poly_k if t_frac <= 0.5 else poly_k1
            if poly_check.contains(_SPt(c[0], c[1])):
                continue

            all_tets.append(slab_global[simplex])

    if not all_tets:
        raise RuntimeError(
            "create_outer_domain_unstructured: no tet elements generated. "
            "Check that cylinder_radius is larger than the BL outer surface."
        )

    elements = np.array(all_tets, dtype=np.int32)

    cyl_labels = list(range(CYL0, CYL0 + N_z * n_cyl_theta))

    return {
        "nodes":    all_nodes_out,
        "elements": elements,
        "sets": {"node": [{"name": "cylinderSurface", "labels": cyl_labels}]},
    }
