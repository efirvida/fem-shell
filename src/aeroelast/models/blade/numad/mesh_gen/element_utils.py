# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:44:35 2023

@author: evans
"""

import numpy as np


def get_el_basis(elType, sVec):
    if elType == "brick8":
        nVec = np.zeros(8, dtype=float)
        dNds = np.zeros((8, 3), dtype=float)
        nVec[0] = -0.125 * (sVec[0] - 1.0) * (sVec[1] - 1.0) * (sVec[2] - 1.0)
        dNds[0, 0] = -0.125 * (sVec[1] - 1.0) * (sVec[2] - 1.0)
        dNds[0, 1] = -0.125 * (sVec[0] - 1.0) * (sVec[2] - 1.0)
        dNds[0, 2] = -0.125 * (sVec[0] - 1.0) * (sVec[1] - 1.0)
        nVec[1] = 0.125 * (sVec[0] + 1.0) * (sVec[1] - 1.0) * (sVec[2] - 1.0)
        dNds[1, 0] = 0.125 * (sVec[1] - 1.0) * (sVec[2] - 1.0)
        dNds[1, 1] = 0.125 * (sVec[0] + 1.0) * (sVec[2] - 1.0)
        dNds[1, 2] = 0.125 * (sVec[0] + 1.0) * (sVec[1] - 1.0)
        nVec[2] = -0.125 * (sVec[0] + 1.0) * (sVec[1] + 1.0) * (sVec[2] - 1.0)
        dNds[2, 0] = -0.125 * (sVec[1] + 1.0) * (sVec[2] - 1.0)
        dNds[2, 1] = -0.125 * (sVec[0] + 1.0) * (sVec[2] - 1.0)
        dNds[2, 2] = -0.125 * (sVec[0] + 1.0) * (sVec[1] + 1.0)
        nVec[3] = 0.125 * (sVec[0] - 1.0) * (sVec[1] + 1.0) * (sVec[2] - 1.0)
        dNds[3, 0] = 0.125 * (sVec[1] + 1.0) * (sVec[2] - 1.0)
        dNds[3, 1] = 0.125 * (sVec[0] - 1.0) * (sVec[2] - 1.0)
        dNds[3, 2] = 0.125 * (sVec[0] - 1.0) * (sVec[1] + 1.0)
        nVec[4] = 0.125 * (sVec[0] - 1.0) * (sVec[1] - 1.0) * (sVec[2] + 1.0)
        dNds[4, 0] = 0.125 * (sVec[1] - 1.0) * (sVec[2] + 1.0)
        dNds[4, 1] = 0.125 * (sVec[0] - 1.0) * (sVec[2] + 1.0)
        dNds[4, 2] = 0.125 * (sVec[0] - 1.0) * (sVec[1] - 1.0)
        nVec[5] = -0.125 * (sVec[0] + 1.0) * (sVec[1] - 1.0) * (sVec[2] + 1.0)
        dNds[5, 0] = -0.125 * (sVec[1] - 1.0) * (sVec[2] + 1.0)
        dNds[5, 1] = -0.125 * (sVec[0] + 1.0) * (sVec[2] + 1.0)
        dNds[5, 2] = -0.125 * (sVec[0] + 1.0) * (sVec[1] - 1.0)
        nVec[6] = 0.125 * (sVec[0] + 1.0) * (sVec[1] + 1.0) * (sVec[2] + 1.0)
        dNds[6, 0] = 0.125 * (sVec[1] + 1.0) * (sVec[2] + 1.0)
        dNds[6, 1] = 0.125 * (sVec[0] + 1.0) * (sVec[2] + 1.0)
        dNds[6, 2] = 0.125 * (sVec[0] + 1.0) * (sVec[1] + 1.0)
        nVec[7] = -0.125 * (sVec[0] - 1.0) * (sVec[1] + 1.0) * (sVec[2] + 1.0)
        dNds[7, 0] = -0.125 * (sVec[1] + 1.0) * (sVec[2] + 1.0)
        dNds[7, 1] = -0.125 * (sVec[0] - 1.0) * (sVec[2] + 1.0)
        dNds[7, 2] = -0.125 * (sVec[0] - 1.0) * (sVec[1] + 1.0)
    elif elType == "wedge6":
        nVec = np.zeros(6, dtype=float)
        dNds = np.zeros((6, 3), dtype=float)
        nVec[0] = -0.5 * (1.0 - sVec[0] - sVec[1]) * (sVec[2] - 1.0)
        dNds[0, 0] = 0.5 * (sVec[2] - 1.0)
        dNds[0, 1] = 0.5 * (sVec[2] - 1.0)
        dNds[0, 2] = -0.5 * (1.0 - sVec[0] - sVec[1])
        nVec[1] = -0.5 * sVec[0] * (sVec[2] - 1.0)
        dNds[1, 0] = -0.5 * (sVec[2] - 1.0)
        dNds[1, 2] = -0.5 * sVec[0]
        nVec[2] = -0.5 * sVec[1] * (sVec[2] - 1.0)
        dNds[2, 1] = -0.5 * (sVec[2] - 1.0)
        dNds[2, 2] = -0.5 * sVec[1]
        nVec[3] = 0.5 * (1.0 - sVec[0] - sVec[1]) * (sVec[2] + 1.0)
        dNds[3, 0] = -0.5 * (sVec[2] + 1.0)
        dNds[3, 1] = -0.5 * (sVec[2] + 1.0)
        dNds[3, 2] = 0.5 * (1.0 - sVec[0] - sVec[1])
        nVec[4] = 0.5 * sVec[0] * (sVec[2] + 1.0)
        dNds[4, 0] = 0.5 * (sVec[2] + 1.0)
        dNds[4, 2] = 0.5 * sVec[0]
        nVec[5] = 0.5 * sVec[1] * (sVec[2] + 1.0)
        dNds[5, 1] = 0.5 * (sVec[2] + 1.0)
        dNds[5, 2] = 0.5 * sVec[1]
    elif elType == "shell4":
        nVec = np.zeros(4, dtype=float)
        dNds = np.zeros((4, 2), dtype=float)
        nVec[0] = 0.25 * (sVec[0] - 1.0) * (sVec[1] - 1.0)
        dNds[0, 0] = 0.25 * (sVec[1] - 1.0)
        dNds[0, 1] = 0.25 * (sVec[0] - 1.0)
        nVec[1] = -0.25 * (sVec[0] + 1.0) * (sVec[1] - 1.0)
        dNds[1, 0] = -0.25 * (sVec[1] - 1.0)
        dNds[1, 1] = -0.25 * (sVec[0] + 1.0)
        nVec[2] = 0.25 * (sVec[0] + 1.0) * (sVec[1] + 1.0)
        dNds[2, 0] = 0.25 * (sVec[1] + 1.0)
        dNds[2, 1] = 0.25 * (sVec[0] + 1.0)
        nVec[3] = -0.25 * (sVec[0] - 1.0) * (sVec[1] + 1.0)
        dNds[3, 0] = -0.25 * (sVec[1] + 1.0)
        dNds[3, 1] = -0.25 * (sVec[0] - 1.0)
    elif elType == "shell3":
        nVec = np.zeros(3, dtype=float)
        dNds = np.zeros((3, 2), dtype=float)
        nVec[0] = 1.0 - sVec[0] - sVec[1]
        dNds[0, 0] = -1.0
        dNds[0, 1] = -1.0
        nVec[1] = sVec[0]
        dNds[1, 0] = 1.0
        nVec[2] = sVec[1]
        dNds[2, 1] = 1.0

    nOut = dict()
    nOut["nVec"] = nVec
    nOut["dNds"] = dNds

    return nOut


def get_element_faces(elType):
    if elType == "tet4":
        faces = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]]
    elif elType == "wedge6":
        faces = [[0, 2, 1], [3, 4, 5], [0, 1, 4, 3], [1, 2, 5, 4], [0, 3, 5, 2]]
    elif elType == "brick8":
        faces = [[3, 2, 1, 0], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    elif elType == "shell3":
        faces = [[0, 1, 2], [0, 2, 1]]
    elif elType == "shell4":
        faces = [[0, 1, 2, 3], [0, 3, 2, 1]]
    return faces


def get_sorted_face_strings(elNds):
    eLen = len(elNds)
    if eLen == 8:
        if elNds[4] == -1:
            faces = get_element_faces("tet4")
        elif elNds[6] == -1:
            faces = get_element_faces("wedge6")
        else:
            faces = get_element_faces("brick8")
    elif eLen == 4:
        if elNds[3] == -1:
            faces = get_element_faces("shell3")
        else:
            faces = get_element_faces("shell4")
    fcStr = list()
    globFc = list()
    for fc in faces:
        glob = list()
        for nd in fc:
            glob.append(elNds[nd])
        globFc.append(glob)
        srted = np.sort(glob)
        fcStr.append(str(srted))
    return fcStr, globFc


def get_el_coord(elNds, ndCrd):
    xCrd = []
    yCrd = []
    zCrd = []
    for nd in elNds:
        if nd > -1:
            xCrd.append(ndCrd[nd][0])
            yCrd.append(ndCrd[nd][1])
            zCrd.append(ndCrd[nd][2])
    elCrd = np.array([xCrd, yCrd, zCrd])
    return elCrd


def get_el_centroid(elCrd):
    elCent = np.zeros(3, dtype=float)
    nnds = len(elCrd[0])
    for ni in range(0, nnds):
        elCent = elCent + elCrd[:, ni]
    elCent = (1.0 / nnds) * elCent
    return elCent


def check_jacobian(elCrd, elType):
    sPts = []
    if elType == "brick8":
        sPts = [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    elif elType == "wedge6":
        sPts = [
            [
                0.0,
                0.0,
                -1.0,
            ],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [
                0.0,
                0.0,
                1.0,
            ],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    for sp in sPts:
        nOut = get_el_basis(elType, sp)
        jac = np.matmul(elCrd, nOut["dNds"])
        det = np.linalg.det(jac)
        if det <= 0.0:
            return False
    return True


def get_volume(elCrd, elType):
    sPts = []
    rt3 = 1.0 / np.sqrt(3)
    op3 = 0.333333333333333333
    if elType == "brick8":
        sPts = rt3 * np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ]
        )
        wt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    elif elType == "wedge6":
        sPts = np.array([[op3, op3, -rt3], [op3, op3, rt3]])
        wt = [0.5, 0.5]
    elif elType == "shell4":
        sPts = rt3 * np.array(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]
        )
        wt = [1.0, 1.0, 1.0, 1.0]
    elif elType == "shell3":
        sPts = np.array([[op3, op3, 0.0], [op3, op3, 0.0]])
        wt = [0.25, 0.25]
    eVol = 0.0
    for i, sp in enumerate(sPts):
        nOut = get_el_basis(elType, sp)
        jac = np.matmul(elCrd, nOut["dNds"])
        if "shell" in elType:
            cp = cross_prod(jac[:, 0], jac[:, 1])
            norm = np.linalg.norm(cp)
            eVol = eVol + norm * wt[i]
        else:
            det = np.linalg.det(jac)
            eVol = eVol + det * wt[i]
    return eVol


def cross_prod(v1, v2):
    cp = np.zeros(3, dtype=float)
    cp[0] = v1[1] * v2[2] - v1[2] * v2[1]
    cp[1] = v1[2] * v2[0] - v1[0] * v2[2]
    cp[2] = v1[0] * v2[1] - v1[1] * v2[0]
    return cp


def rotate_orient(oriMat, rotVec):
    mag = np.linalg.norm(rotVec)
    if mag > 0.0001:
        a1 = np.zeros((3, 3), dtype=float)
        sth = np.sin(mag)
        cth = np.cos(mag)

        unitRot = (1.0 / mag) * rotVec
        a1[0] = unitRot

        i1 = 0
        if abs(unitRot[1]) < abs(unitRot[0]):
            i1 = 1
        if abs(unitRot[2]) < abs(unitRot[i1]):
            i1 = 2

        a1[1, i1] = np.sqrt(1.0 - unitRot[i1] * unitRot[i1])

        for i2 in range(0, 3):
            if i2 != i1:
                a1[1, i2] = -a1[0, i1] * a1[0, i2] / a1[1, i1]

        a1[2] = cross_prod(a1[0], a1[1])

        a2 = np.array([[1.0, 0.0, 0.0], [0.0, cth, -sth], [0.0, sth, cth]])

        a3 = np.matmul(a2, a1)
        a3T = np.transpose(a3)
        a4 = np.matmul(a3T, a1)

        return np.matmul(oriMat, a4)
    else:
        return oriMat


def correct_orient(aprxOri, elCrd, elType):
    if elType == "shell4":
        v0 = elCrd[:, 2] - elCrd[:, 0]
        v1 = elCrd[:, 3] - elCrd[:, 1]
    elif elType == "shell3":
        v0 = elCrd[:, 1] - elCrd[:, 0]
        v1 = elCrd[:, 2] - elCrd[:, 0]
    else:
        pstr = "Warning: element type " + elType + " not supported for correctOrient"
        print(pstr)
        return aprxOri
    v2 = cross_prod(v0, v1)
    mag = np.linalg.norm(v2)
    v2 = (1.0 / mag) * v2
    a2 = aprxOri[2]
    dp = np.dot(v2, a2)
    if dp < 0.0:
        v2 = -1.0 * v2
    cp = cross_prod(a2, v2)
    mag = np.linalg.norm(cp)
    theta = np.arcsin(mag)
    rotVec = (theta / mag) * cp
    return rotate_orient(aprxOri, rotVec)


def spt_in_el(elType, sPt):
    ## sPt is the element space vector from the centroid of the element
    if elType == "brick8":
        if abs(sPt[0]) > 1.0:
            return False
        if abs(sPt[1]) > 1.0:
            return False
        if abs(sPt[2]) > 1.0:
            return False
        return True
    if elType == "wedge6":
        sO = sPt + 0.333333333333333
        if sO[0] < 0.0:
            return False
        if sO[1] < 0.0:
            return False
        if sO[1] > (1.0 - sO[0]):
            return False
        if abs(sPt[2]) > 1.0:
            return False
        return True
    if elType == "shell4":
        if abs(sPt[0]) > 1.0:
            return False
        if abs(sPt[1]) > 1.0:
            return False
        return True
    if elType == "shell3":
        sO = sPt + 0.333333333333333
        if sO[0] < 0.0:
            return False
        if sO[1] < 0.0:
            return False
        if sO[1] > (1.0 - sO[0]):
            return False
        return True
    return False


def get_proj_dist(elCrd, elType, ptCrd):
    elCent = get_el_centroid(elCrd)
    xDist = ptCrd - elCent
    sVec = np.zeros(3, dtype=float)
    nOut = get_el_basis(elType, sVec)
    dxds = np.matmul(elCrd, nOut["dNds"])
    Q, R = np.linalg.qr(dxds)
    qDis = np.matmul(xDist, Q)
    sVec = np.linalg.solve(R, qDis)
    if spt_in_el(elType, sVec):
        if elType in "brick8 wedge6":
            dist = 0.0
        else:
            xVec = np.matmul(dxds, sVec)
            xDist = xDist - xVec
            dist = np.linalg.norm(xDist)
    else:
        fact = 0.5
        while fact < 0.999:
            while not spt_in_el(elType, sVec):
                sVec = fact * sVec
            sVec = (1.0 / fact) * sVec
            fact = np.sqrt(fact)
        xVec = np.matmul(dxds, sVec)
        xDist = xDist - xVec
        dist = np.linalg.norm(xDist)
    if elType == "wedge6":
        sVec[0:2] = sVec[0:2] + 0.333333333333333
    elif elType == "shell3":
        sVec = sVec + 0.333333333333333
    nOut = get_el_basis(elType, sVec)
    projOut = dict()
    projOut["distance"] = dist
    projOut["nVec"] = nOut["nVec"]
    return projOut


def get_solid_surf_proj(elCrd, elType, ptCrd):
    if elType == "brick8":
        nnds = 8
        faces = [[0, 4, 7, 3], [1, 2, 6, 5], [0, 1, 5, 4], [3, 7, 6, 2], [0, 3, 2, 1], [4, 5, 6, 7]]
    elif elType == "wedge6":
        nnds = 6
        faces = [[0, 2, 5, 3], [0, 3, 4, 1], [1, 4, 5, 2], [0, 1, 2], [3, 5, 4]]
    else:
        faces = []
        pstr = "Warning: element type " + elType + " not currently supported in getSolidSurfProj"
        print(pstr)
    minDist = 1.0e100
    minFace = -1
    minPO = dict()
    fi = 0
    for fc in faces:
        xC = []
        yC = []
        zC = []
        for nd in fc:
            xC.append(elCrd[0, nd])
            yC.append(elCrd[1, nd])
            zC.append(elCrd[2, nd])
        shCrd = np.array([xC, yC, zC])
        if len(xC) == 4:
            pO = get_proj_dist(shCrd, "shell4", ptCrd)
        elif len(xC) == 3:
            pO = get_proj_dist(shCrd, "shell3", ptCrd)
        if pO["distance"] < minDist:
            minDist = pO["distance"]
            minFace = fi
            minPO = pO
        fi = fi + 1
    shNVec = minPO["nVec"]
    nVec = np.zeros(nnds, dtype=float)
    shni = 0
    for ni in faces[minFace]:
        nVec[ni] = shNVec[shni]
        shni = shni + 1
    projOut = dict()
    projOut["distance"] = minDist
    projOut["nVec"] = nVec
    return projOut


def get_vertex_normals(nodes, elements):
    """Compute outward-pointing vertex normals for a quad4 surface mesh.

    The sign is corrected per element — before accumulation — by comparing
    each element's face normal to the vector from the cross-section centroid
    to the element centroid.  This is robust for any winding order and works
    correctly at the trailing-edge panel, where the element centroid is at the
    extreme aft position of the section (so the outward direction is
    unambiguous even for a blunt-TE narrow panel).

    Parameters
    ----------
    nodes : ndarray, shape (N, 3)
        Node coordinates [x, y, z].
    elements : ndarray, shape (M, 4)
        Quad4 connectivity, 0-based local indices.  Elements whose
        fourth index is -1 (degenerate triangles) are skipped.

    Returns
    -------
    normals : ndarray, shape (N, 3)
        Unit outward normals at every node.
    """
    N = len(nodes)
    weighted = np.zeros((N, 3), dtype=float)

    # Precompute section centroid (XY) for every node indexed by Z-level.
    # Using a rounded integer key avoids floating-point equality issues.
    z_coords = nodes[:, 2]
    z_span = z_coords.max() - z_coords.min()
    tol = 1e-6 * z_span if z_span > 0 else 1e-6
    z_int = np.round(z_coords / tol).astype(np.int64)
    unique_z_int = np.unique(z_int)

    # node_sx[i], node_sy[i] = XY centroid of the section that contains node i
    node_sx = np.empty(N, dtype=float)
    node_sy = np.empty(N, dtype=float)
    for zi in unique_z_int:
        mask = z_int == zi
        cx = nodes[mask, 0].mean()
        cy = nodes[mask, 1].mean()
        node_sx[mask] = cx
        node_sy[mask] = cy

    # Detect trailing-edge closure panels and skip them from normal
    # accumulation.  TE panels are very thin (TE bluntness << element size)
    # so their minimum edge length is far below the typical mesh size.
    # We use an adaptive threshold: the 3rd-percentile of all min-edge values
    # × 10.  This sits cleanly in the gap between TE panels (< ~0.02 m) and
    # normal surface panels (> ~0.05 m) for the IEA-15-240-RWT at 0.15 m.
    quad_els = np.array(elements)
    is_quad = quad_els[:, 3] != -1
    # For quad elements, compute the four edge lengths vectorised
    p0 = nodes[np.where(is_quad, quad_els[:, 0], 0)]
    p1 = nodes[np.where(is_quad, quad_els[:, 1], 0)]
    p2 = nodes[np.where(is_quad, quad_els[:, 2], 0)]
    p3 = nodes[np.where(is_quad, quad_els[:, 3], 0)]
    e01 = np.linalg.norm(p1 - p0, axis=1)
    e12 = np.linalg.norm(p2 - p1, axis=1)
    e23 = np.linalg.norm(p3 - p2, axis=1)
    e30 = np.linalg.norm(p0 - p3, axis=1)
    min_edges = np.where(is_quad, np.minimum(np.minimum(e01, e12), np.minimum(e23, e30)), np.inf)
    # Threshold = 10× the 1st-percentile of valid (quad) min-edges.
    # p1 lands in the TE-panel range (< ~0.02 m for a 0.15 m mesh).
    # × 10 puts the cut-off right in the gap (0.02 – 0.05 m) between
    # the thin TE closure panels and the normal surface panels.
    # Using p3 × 10 was too aggressive (0.34 m > any edge → 100% filtered).
    finite_me = min_edges[np.isfinite(min_edges)]
    te_threshold = np.percentile(finite_me, 1) * 10.0 if len(finite_me) else 0.0

    for ei, el in enumerate(elements):
        if el[3] == -1:
            continue
        # Skip TE closure panels: their thin edge contaminates the normals of
        # adjacent nodes, pulling the BL extrusion toward aft instead of outward.
        if min_edges[ei] < te_threshold:
            continue
        n0, n1, n2, n3 = el[0], el[1], el[2], el[3]
        p0, p1, p2, p3 = nodes[n0], nodes[n1], nodes[n2], nodes[n3]

        d1 = p2 - p0
        d2 = p3 - p1
        raw = np.cross(d1, d2)
        mag = np.linalg.norm(raw)
        if mag < 1e-14:
            continue
        unit = raw / mag

        # Outward direction for this element: from the (averaged) section
        # centroid to the element centroid in XY.  Using the element's own
        # section centroids (averaged over its 4 corner nodes) gives the
        # correct reference even when nodes span two Z-levels (span-wise
        # elements) or sit at the extreme TE position.
        el_cx = (p0[0] + p1[0] + p2[0] + p3[0]) * 0.25
        el_cy = (p0[1] + p1[1] + p2[1] + p3[1]) * 0.25
        sc_cx = (node_sx[n0] + node_sx[n1] + node_sx[n2] + node_sx[n3]) * 0.25
        sc_cy = (node_sy[n0] + node_sy[n1] + node_sy[n2] + node_sy[n3]) * 0.25
        out_x = el_cx - sc_cx
        out_y = el_cy - sc_cy

        # Flip if the face normal opposes the outward direction
        if unit[0] * out_x + unit[1] * out_y < 0:
            unit = -unit

        for ni in (n0, n1, n2, n3):
            weighted[ni] += unit

    # Normalise — no per-node sign flip needed; sign is already correct
    mags = np.linalg.norm(weighted, axis=1, keepdims=True)
    mags = np.where(mags < 1e-14, 1.0, mags)
    return weighted / mags
