import numpy as np

##from pynumad.mesh_gen.shellClasses import shellRegion, elementSet, NuMesh3D, spatialGridList2D, spatialGridList3D
from fem_shell.models.blade.numad.mesh_gen.boundary2d import *
from fem_shell.models.blade.numad.mesh_gen.element_utils import *
from fem_shell.models.blade.numad.mesh_gen.mesh2d import *
from fem_shell.models.blade.numad.mesh_gen.mesh_tools import *
from fem_shell.models.blade.numad.mesh_gen.surface import Surface
from fem_shell.models.blade.numad.utils.interpolation import interpolator_wrap


def get_shell_mesh(blade, elementSize):
    """
    This method generates a finite element shell mesh for the blade, based on what is
    stored in blade.geometry.coordinates, blade.keypoints.key_points,
    and blade.geometry.profiles.  Output is given as a python dictionary.

    Parameters
    -----------
    blade: Blade
    forSolid: bool
    elementSize: float

    Returns
    -------
    meshData:
    Nodes and elements for outer shell and shear webs:
    nodes:
        - [x, y, z]
        - [x, y, z]
        ...
    elements:
        - [n1,n2,n3,n4]
        - [n1,n2,n3,n4]
        ...
    Set list and section list for the outer shell and shear webs.
    These are companion lists with the same length and order,
    so meshData['sets']['element'][i] corresponds to meshData['sections'][i]
    sets:
        element:
            - name: set1Name
              labels: [e1, e2, e3 ...]
            - name: set2Name
              labels: [e1, e2, e3 ...]
            ...
    sections:
        - type: 'shell'
          layup:
             - [materialid,thickness,angle] ## layer 1
             - [materialid,thickness,angle] ## layer 2
             ...
          elementSet: set1Name
        - type: 'shell'
          layup:
             - [materialid,thickness,angle] ## layer 1
             - [materialid,thickness,angle] ## layer 2
             ...
          elementSet: set2Name
    """
    geometry = blade.geometry
    coordinates = geometry.coordinates
    profiles = geometry.profiles
    key_points = blade.keypoints.key_points
    stacks = blade.stackdb.stacks
    swstacks = blade.stackdb.swstacks

    geomSz = coordinates.shape
    lenGeom = geomSz[0]
    numXsec = geomSz[2]
    XSCurvePts = np.array([], dtype=int)

    ## Determine the key curve points along the OML at each cross section
    for i in range(numXsec):
        keyPts = np.array([0])
        minDist = 1
        lePt = 0
        for j in range(lenGeom):
            prof = profiles[j, :, i]
            mag = np.linalg.norm(prof)
            if mag < minDist:
                minDist = mag
                lePt = j

        for j in range(5):
            kpCrd = key_points[j, :, i]
            minDist = geometry.ichord[i]
            pti = 1
            for k in range(lePt):
                ptCrd = coordinates[k, :, i]
                vec = ptCrd - kpCrd
                mag = np.linalg.norm(vec)
                if mag < minDist:
                    minDist = mag
                    pti = k
            keyPts = np.concatenate((keyPts, [pti]))
            coordinates[pti, :, i] = np.array(kpCrd)

        keyPts = np.concatenate((keyPts, [lePt]))
        for j in range(5, 10):
            kpCrd = key_points[j, :, i]
            minDist = geometry.ichord[i]
            pti = 1
            for k in range(lePt, lenGeom):
                ptCrd = coordinates[k, :, i]
                vec = ptCrd - kpCrd
                mag = np.linalg.norm(vec)
                if mag < minDist:
                    minDist = mag
                    pti = k
            keyPts = np.concatenate((keyPts, [pti]))
            coordinates[pti, :, i] = np.array(kpCrd)

        keyPts = np.concatenate((keyPts, [lenGeom - 1]))
        allPts = np.array([keyPts[0]])
        for j in range(0, len(keyPts) - 1):
            secPts = np.linspace(keyPts[j], keyPts[j + 1], 4)
            secPts = np.round(secPts).astype(int)
            allPts = np.concatenate((allPts, secPts[1:4]))

        XSCurvePts = np.vstack((XSCurvePts, allPts)) if XSCurvePts.size else allPts
    rws, cls = XSCurvePts.shape

    ## Create longitudinal splines down the blade through each of the key X-section points

    splineX = coordinates[XSCurvePts[0, :], 0, 0]
    splineY = coordinates[XSCurvePts[0, :], 1, 0]
    splineZ = coordinates[XSCurvePts[0, :], 2, 0]
    for i in range(1, rws):
        Xrow = coordinates[XSCurvePts[i, :], 0, i]
        splineX = np.vstack((splineX, Xrow.T))
        Yrow = coordinates[XSCurvePts[i, :], 1, i]
        splineY = np.vstack((splineY, Yrow.T))
        Zrow = coordinates[XSCurvePts[i, :], 2, i]
        splineZ = np.vstack((splineZ, Zrow.T))

    spParam = np.transpose(np.linspace(0, 1, rws))
    nSpi = rws + 2 * (rws - 1)
    spParami = np.transpose(np.linspace(0, 1, nSpi))
    splineXi = interpolator_wrap(spParam, splineX[:, 0], spParami, "pchip")
    splineYi = interpolator_wrap(spParam, splineY[:, 0], spParami, "pchip")
    splineZi = interpolator_wrap(spParam, splineZ[:, 0], spParami, "pchip")
    for i in range(1, cls):
        splineXi = np.vstack([
            splineXi,
            interpolator_wrap(spParam, splineX[:, i], spParami, "pchip"),
        ])
        splineYi = np.vstack([
            splineYi,
            interpolator_wrap(spParam, splineY[:, i], spParami, "pchip"),
        ])
        splineZi = np.vstack([
            splineZi,
            interpolator_wrap(spParam, splineZ[:, i], spParami, "pchip"),
        ])
    splineXi = splineXi.T
    splineYi = splineYi.T
    splineZi = splineZi.T

    ## Generate the mesh using the splines as surface guides

    ## --------------------------TEMP
    ## secNel = [1,3,15,5,8,1,1,13,5,12,3,1]
    ## secNel = [1,4,10,10,8,2,2,8,10,10,4,1]
    ## --------------------------END TEMP

    bladeSurf = Surface()
    ## Outer shell sections
    outShES = set()
    secList = []
    stPt = 0
    for i in range(rws - 1):
        # if stPt < frstXS:
        #     stSec = 0
        #     endSec = 11
        #     stSp = 0
        # else:
        #     stSec = 1
        #     endSec = 10
        #     stSp = 3
        stSec = 0
        endSec = 11
        stSp = 0
        for j in range(stSec, endSec + 1):
            shellKp = np.array([
                [splineXi[stPt, stSp], splineYi[stPt, stSp], splineZi[stPt, stSp]],
                [
                    splineXi[stPt, stSp + 3],
                    splineYi[stPt, stSp + 3],
                    splineZi[stPt, stSp + 3],
                ],
                [
                    splineXi[stPt + 3, stSp + 3],
                    splineYi[stPt + 3, stSp + 3],
                    splineZi[stPt + 3, stSp + 3],
                ],
                [
                    splineXi[stPt + 3, stSp],
                    splineYi[stPt + 3, stSp],
                    splineZi[stPt + 3, stSp],
                ],
                [
                    splineXi[stPt, stSp + 1],
                    splineYi[stPt, stSp + 1],
                    splineZi[stPt, stSp + 1],
                ],
                [
                    splineXi[stPt, stSp + 2],
                    splineYi[stPt, stSp + 2],
                    splineZi[stPt, stSp + 2],
                ],
                [
                    splineXi[stPt + 1, stSp + 3],
                    splineYi[stPt + 1, stSp + 3],
                    splineZi[stPt + 1, stSp + 3],
                ],
                [
                    splineXi[stPt + 2, stSp + 3],
                    splineYi[stPt + 2, stSp + 3],
                    splineZi[stPt + 2, stSp + 3],
                ],
                [
                    splineXi[stPt + 3, stSp + 2],
                    splineYi[stPt + 3, stSp + 2],
                    splineZi[stPt + 3, stSp + 2],
                ],
                [
                    splineXi[stPt + 3, stSp + 1],
                    splineYi[stPt + 3, stSp + 1],
                    splineZi[stPt + 3, stSp + 1],
                ],
                [
                    splineXi[stPt + 2, stSp],
                    splineYi[stPt + 2, stSp],
                    splineZi[stPt + 2, stSp],
                ],
                [
                    splineXi[stPt + 1, stSp],
                    splineYi[stPt + 1, stSp],
                    splineZi[stPt + 1, stSp],
                ],
                [
                    splineXi[stPt + 1, stSp + 1],
                    splineYi[stPt + 1, stSp + 1],
                    splineZi[stPt + 1, stSp + 1],
                ],
                [
                    splineXi[stPt + 1, stSp + 2],
                    splineYi[stPt + 1, stSp + 2],
                    splineZi[stPt + 1, stSp + 2],
                ],
                [
                    splineXi[stPt + 2, stSp + 2],
                    splineYi[stPt + 2, stSp + 2],
                    splineZi[stPt + 2, stSp + 2],
                ],
                [
                    splineXi[stPt + 2, stSp + 1],
                    splineYi[stPt + 2, stSp + 1],
                    splineZi[stPt + 2, stSp + 1],
                ],
            ])
            # vec = shellKp[1, :] - shellKp[0, :]
            # mag = np.linalg.norm(vec)
            # nEl = np.array([], dtype=int)
            # nEl = np.concatenate([nEl, [np.ceil(mag / elementSize).astype(int)]])
            # vec = shellKp[2, :] - shellKp[1, :]
            # mag = np.linalg.norm(vec)
            # nEl = np.concatenate([nEl, [np.ceil(mag / elementSize).astype(int)]])
            # vec = shellKp[3, :] - shellKp[2, :]
            # mag = np.linalg.norm(vec)
            # nEl = np.concatenate([nEl, [np.ceil(mag / elementSize).astype(int)]])
            # vec = shellKp[0, :] - shellKp[3, :]
            # mag = np.linalg.norm(vec)
            # nEl = np.concatenate([nEl, [np.ceil(mag / elementSize).astype(int)]])

            vec = shellKp[1, :] - shellKp[0, :]
            mag = np.linalg.norm(vec)

            nEl1 = int(np.ceil(mag / elementSize))
            ##
            ## nEl1 = secNel[j]
            ##

            vec = shellKp[2, :] - shellKp[1, :]
            mag = np.linalg.norm(vec)
            nEl2 = int(np.ceil(mag / elementSize))
            vec = shellKp[3, :] - shellKp[2, :]
            mag = np.linalg.norm(vec)

            nEl3 = int(np.ceil(mag / elementSize))
            ##
            ## nEl3 = secNel[j]
            ##

            vec = shellKp[0, :] - shellKp[3, :]
            mag = np.linalg.norm(vec)
            nEl4 = int(np.ceil(mag / elementSize))
            nEl = np.array([nEl1, nEl2, nEl3, nEl4])

            bladeSurf.addShellRegion(
                "quad3",
                shellKp,
                nEl,
                name=stacks[j, i].name,
                elType="quad",
                meshMethod="structured",
            )
            outShES.add(stacks[j, i].name)
            newSec = {}
            newSec["type"] = "shell"
            layup = []
            for pg in stacks[j, i].plygroups:
                totThick = 0.001 * pg.thickness * pg.nPlies
                ply = [pg.materialid, totThick, pg.angle]
                layup.append(ply)
            newSec["layup"] = layup
            newSec["elementSet"] = stacks[j, i].name
            newSec["xDir"] = (shellKp[3, :] - shellKp[0, :]) + (shellKp[2, :] - shellKp[1, :])
            newSec["xyDir"] = (shellKp[1, :] - shellKp[0, :]) + (shellKp[2, :] - shellKp[3, :])
            secList.append(newSec)
            stSp = stSp + 3
        stPt = stPt + 3

    ## Shear web sections
    swES = set()
    stPt = 0
    web1Sets = np.array([])
    web2Sets = np.array([])
    for i in range(rws - 1):
        if swstacks[0][i].plygroups:
            shellKp = np.zeros((16, 3))
            shellKp[0, :] = np.array([splineXi[stPt, 12], splineYi[stPt, 12], splineZi[stPt, 12]])
            shellKp[1, :] = np.array([splineXi[stPt, 24], splineYi[stPt, 24], splineZi[stPt, 24]])
            shellKp[2, :] = np.array([
                splineXi[stPt + 3, 24],
                splineYi[stPt + 3, 24],
                splineZi[stPt + 3, 24],
            ])
            shellKp[3, :] = np.array([
                splineXi[stPt + 3, 12],
                splineYi[stPt + 3, 12],
                splineZi[stPt + 3, 12],
            ])
            shellKp[6, :] = np.array([
                splineXi[stPt + 1, 24],
                splineYi[stPt + 1, 24],
                splineZi[stPt + 1, 24],
            ])
            shellKp[7, :] = np.array([
                splineXi[stPt + 2, 24],
                splineYi[stPt + 2, 24],
                splineZi[stPt + 2, 24],
            ])
            shellKp[10, :] = np.array([
                splineXi[stPt + 2, 12],
                splineYi[stPt + 2, 12],
                splineZi[stPt + 2, 12],
            ])
            shellKp[11, :] = np.array([
                splineXi[stPt + 1, 12],
                splineYi[stPt + 1, 12],
                splineZi[stPt + 1, 12],
            ])
            shellKp[4, :] = 0.6666 * shellKp[0, :] + 0.3333 * shellKp[1, :]
            shellKp[5, :] = 0.3333 * shellKp[0, :] + 0.6666 * shellKp[1, :]
            shellKp[8, :] = 0.6666 * shellKp[2, :] + 0.3333 * shellKp[3, :]
            shellKp[9, :] = 0.3333 * shellKp[2, :] + 0.6666 * shellKp[3, :]
            shellKp[12, :] = 0.6666 * shellKp[11, :] + 0.3333 * shellKp[6, :]
            shellKp[13, :] = 0.3333 * shellKp[11, :] + 0.6666 * shellKp[6, :]
            shellKp[14, :] = 0.6666 * shellKp[7, :] + 0.3333 * shellKp[10, :]
            shellKp[15, :] = 0.3333 * shellKp[7, :] + 0.6666 * shellKp[10, :]

            # vec = shellKp[1, :] - shellKp[0, :]
            # mag = np.linalg.norm(vec)

            # nEl = np.array([], dtype=int)
            # nEl = np.concatenate([nEl, [np.ceil(mag / elementSize).astype(int)]])
            # vec = shellKp[2, :] - shellKp[1, :]
            # mag = np.linalg.norm(vec)
            # nEl = np.concatenate([nEl, [np.ceil(mag / elementSize).astype(int)]])
            # vec = shellKp[3, :] - shellKp[2, :]
            # mag = np.linalg.norm(vec)
            # nEl = np.concatenate([nEl, [np.ceil(mag / elementSize).astype(int)]])
            # vec = shellKp[0, :] - shellKp[3, :]
            # mag = np.linalg.norm(vec)
            # nEl = np.concatenate([nEl, [np.ceil(mag / elementSize).astype(int)]])

            vec = shellKp[1, :] - shellKp[0, :]
            mag = np.linalg.norm(vec)

            nEl1 = int(np.ceil(mag / elementSize))
            ##
            ## nEl1 = 8
            ##

            vec = shellKp[2, :] - shellKp[1, :]
            mag = np.linalg.norm(vec)
            nEl2 = int(np.ceil(mag / elementSize))
            vec = shellKp[3, :] - shellKp[2, :]
            mag = np.linalg.norm(vec)

            nEl3 = int(np.ceil(mag / elementSize))
            ##
            ## nEl3 = 8
            ##

            vec = shellKp[0, :] - shellKp[3, :]
            mag = np.linalg.norm(vec)
            nEl4 = int(np.ceil(mag / elementSize))
            nEl = np.array([nEl1, nEl2, nEl3, nEl4])

            bladeSurf.addShellRegion(
                "quad3",
                shellKp,
                nEl,
                name=swstacks[0][i].name,
                elType="quad",
                meshMethod="structured",
            )
            swES.add(swstacks[0][i].name)
            newSec = {}
            newSec["type"] = "shell"
            layup = []
            for pg in swstacks[0][i].plygroups:
                totThick = 0.001 * pg.thickness * pg.nPlies
                ply = [pg.materialid, totThick, pg.angle]
                layup.append(ply)
            newSec["layup"] = layup
            newSec["elementSet"] = swstacks[0][i].name
            newSec["xDir"] = np.array([0.0, 0.0, 1.0])
            newSec["xyDir"] = (shellKp[1, :] - shellKp[0, :]) + (shellKp[2, :] - shellKp[3, :])
            secList.append(newSec)
        if swstacks[1][i].plygroups:
            shellKp = np.zeros((16, 3))
            shellKp[0, :] = np.array([splineXi[stPt, 27], splineYi[stPt, 27], splineZi[stPt, 27]])
            shellKp[1, :] = np.array([splineXi[stPt, 9], splineYi[stPt, 9], splineZi[stPt, 9]])
            shellKp[2, :] = np.array([
                splineXi[stPt + 3, 9],
                splineYi[stPt + 3, 9],
                splineZi[stPt + 3, 9],
            ])
            shellKp[3, :] = np.array([
                splineXi[stPt + 3, 27],
                splineYi[stPt + 3, 27],
                splineZi[stPt + 3, 27],
            ])
            shellKp[6, :] = np.array([
                splineXi[stPt + 1, 9],
                splineYi[stPt + 1, 9],
                splineZi[stPt + 1, 9],
            ])
            shellKp[7, :] = np.array([
                splineXi[stPt + 2, 9],
                splineYi[stPt + 2, 9],
                splineZi[stPt + 2, 9],
            ])
            shellKp[10, :] = np.array([
                splineXi[stPt + 2, 27],
                splineYi[stPt + 2, 27],
                splineZi[stPt + 2, 27],
            ])
            shellKp[11, :] = np.array([
                splineXi[stPt + 1, 27],
                splineYi[stPt + 1, 27],
                splineZi[stPt + 1, 27],
            ])
            shellKp[4, :] = 0.6666 * shellKp[0, :] + 0.3333 * shellKp[1, :]
            shellKp[5, :] = 0.3333 * shellKp[0, :] + 0.6666 * shellKp[1, :]
            shellKp[8, :] = 0.6666 * shellKp[2, :] + 0.3333 * shellKp[3, :]
            shellKp[9, :] = 0.3333 * shellKp[2, :] + 0.6666 * shellKp[3, :]
            shellKp[12, :] = 0.6666 * shellKp[11, :] + 0.3333 * shellKp[6, :]
            shellKp[13, :] = 0.3333 * shellKp[11, :] + 0.6666 * shellKp[6, :]
            shellKp[14, :] = 0.6666 * shellKp[7, :] + 0.3333 * shellKp[10, :]
            shellKp[15, :] = 0.3333 * shellKp[7, :] + 0.6666 * shellKp[10, :]

            # vec = shellKp[1, :] - shellKp[0, :]
            # mag = np.linalg.norm(vec)

            # nEl = np.array([], dtype=int)
            # nEl = np.concatenate([nEl, [np.ceil(mag / elementSize).astype(int)]])
            # vec = shellKp[2, :] - shellKp[1, :]
            # mag = np.linalg.norm(vec)
            # nEl = np.concatenate([nEl, [np.ceil(mag / elementSize).astype(int)]])
            # vec = shellKp[3, :] - shellKp[2, :]
            # mag = np.linalg.norm(vec)
            # nEl = np.concatenate([nEl, [np.ceil(mag / elementSize).astype(int)]])
            # vec = shellKp[0, :] - shellKp[3, :]
            # mag = np.linalg.norm(vec)
            # nEl = np.concatenate([nEl, [np.ceil(mag / elementSize).astype(int)]])

            vec = shellKp[1, :] - shellKp[0, :]
            mag = np.linalg.norm(vec)

            nEl1 = int(np.ceil(mag / elementSize))
            ##
            ## nEl1 = 8
            ##

            vec = shellKp[2, :] - shellKp[1, :]
            mag = np.linalg.norm(vec)
            nEl2 = int(np.ceil(mag / elementSize))
            vec = shellKp[3, :] - shellKp[2, :]
            mag = np.linalg.norm(vec)

            nEl3 = int(np.ceil(mag / elementSize))
            ##
            ## nEl3 = 8
            ##

            vec = shellKp[0, :] - shellKp[3, :]
            mag = np.linalg.norm(vec)
            nEl4 = int(np.ceil(mag / elementSize))
            nEl = np.array([nEl1, nEl2, nEl3, nEl4])

            bladeSurf.addShellRegion(
                "quad3",
                shellKp,
                nEl,
                name=swstacks[1][i].name,
                elType="quad",
                meshMethod="structured",
            )
            swES.add(swstacks[1][i].name)
            newSec = {}
            newSec["type"] = "shell"
            layup = []
            for pg in swstacks[1][i].plygroups:
                totThick = 0.001 * pg.thickness * pg.nPlies
                ply = [pg.materialid, totThick, pg.angle]
                layup.append(ply)
            newSec["layup"] = layup
            newSec["elementSet"] = swstacks[1][i].name
            newSec["xDir"] = np.array([0.0, 0.0, 1.0])
            newSec["xyDir"] = (shellKp[1, :] - shellKp[0, :]) + (shellKp[2, :] - shellKp[3, :])
            secList.append(newSec)
        stPt = stPt + 3

    ## Generate Shell mesh

    # print("getting blade mesh")
    shellData = bladeSurf.getSurfaceMesh()
    shellData["sections"] = secList

    ## Get local direction cosine orientations for individual elements
    # print("getting element orientations")
    nodes = shellData["nodes"]
    elements = shellData["elements"]
    numEls = len(shellData["elements"])
    elOri = np.zeros((numEls, 9), dtype=float)

    esi = 0
    for sec in secList:
        dirCos = get_direction_cosines(sec["xDir"], sec["xyDir"])
        es = shellData["sets"]["element"][esi]
        for ei in es["labels"]:
            eX = []
            eY = []
            eZ = []
            for ndi in elements[ei]:
                if ndi > -1:
                    eX.append(nodes[ndi, 0])
                    eY.append(nodes[ndi, 1])
                    eZ.append(nodes[ndi, 2])
            if len(eX) == 3:
                elType = "shell3"
            else:
                elType = "shell4"
            elCrd = np.array([eX, eY, eZ])
            elDirCos = correct_orient(dirCos, elCrd, elType)
            elOri[ei, 0:3] = elDirCos[0]
            elOri[ei, 3:6] = elDirCos[1]
            elOri[ei, 6:9] = elDirCos[2]
        esi = esi + 1

    shellData["elementOrientations"] = elOri

    ## Get all outer shell and all shear web element sets

    outerLab = []
    swLab = []
    for es in shellData["sets"]["element"]:
        nm = es["name"]
        if nm in outShES:
            outerLab.extend(es["labels"])
        elif nm in swES:
            swLab.extend(es["labels"])
    outerSet = {}
    outerSet["name"] = "allOuterShellEls"
    outerSet["labels"] = outerLab
    shellData["sets"]["element"].append(outerSet)
    swSet = {}
    swSet["name"] = "allShearWebEls"
    swSet["labels"] = swLab
    shellData["sets"]["element"].append(swSet)

    ## Get root (Zmin) node set
    minZ = np.min(splineZi)
    rootLabs = []
    lab = 0
    for nd in nodes:
        if np.abs(nd[2] - minZ) < 0.25 * elementSize:
            rootLabs.append(lab)
        lab = lab + 1
    newSet = {}
    newSet["name"] = "RootNodes"
    newSet["labels"] = rootLabs
    try:
        shellData["sets"]["node"].append(newSet)
    except:
        nodeSets = []
        nodeSets.append(newSet)
        shellData["sets"]["node"] = nodeSets

    matList = []
    for mn in blade.definition.materials:
        newMat = {}
        mat = blade.definition.materials[mn]
        newMat["name"] = mat.name
        newMat["density"] = mat.density
        newMat["elastic"] = {}
        newMat["elastic"]["E"] = [mat.ex, mat.ey, mat.ez]
        if mat.type == "isotropic":
            nu = mat.prxy
            newMat["elastic"]["nu"] = [nu, nu, nu]
        else:
            newMat["elastic"]["nu"] = [mat.prxy, mat.prxz, mat.pryz]
        newMat["elastic"]["G"] = [mat.gxy, mat.gxz, mat.gyz]
        matList.append(newMat)

    shellData["materials"] = matList

    return shellData
