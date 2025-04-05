########################################################################
#                    Part of the SNL NuMAD Toolbox                     #
#  Developed by Sandia National Laboratories Wind Energy Technologies  #
#              See license.txt for disclaimer information              #
########################################################################

import numpy as np
import yaml

from pynumad.objects.Airfoil import Airfoil
from pynumad.objects.Component import Component
from pynumad.objects.Material import Material
from pynumad.utils.interpolation import interpolator_wrap
from pynumad.utils.misc_utils import _parse_data


def yaml_to_blade(blade, filename: str, write_airfoils: bool = False):
    """
    This method writes blade information from a .yaml file to a Blade object.
    The yaml file is expected to be formatted according to the WindIO ontology.
    See https://windio.readthedocs.io/en/stable/source/turbine.html.

    Parameters
    ----------
    blade : Blade
    filename : string
        path to .yaml file
    write_airfoils : bool
        Set true to write airfoil files while reading in data. Defaults to false.

    Returns
    -------
    blade : Blade
        input blade object populated with yaml data
    """

    # Read in yaml file as a nested dictionary
    with open(filename) as blade_yaml:
        # data = yaml.load(blade_yaml,Loader=yaml.FullLoader)
        data = yaml.load(blade_yaml, Loader=yaml.Loader)

    # Name some key subdata
    blade_outer_shape_bem = data["components"]["blade"]["outer_shape_bem"]

    # older versions of wind ontology do not have 'outer_shape_bem' subsection for hub data
    try:
        hub_outer_shape_bem = data["components"]["hub"]["outer_shape_bem"]
    except KeyError:
        hub_outer_shape_bem = data["components"]["hub"]

    blade_internal_structure = data["components"]["blade"]["internal_structure_2d_fem"]
    af_data = data["airfoils"]
    mat_data = data["materials"]

    ### STATIONS / AIRFOILS
    _add_stations(
        blade, blade_outer_shape_bem, hub_outer_shape_bem, af_data, filename, write_airfoils
    )

    ### MATERIALS
    _add_materials(blade, mat_data)

    # Update "grid" and "values" keys to cover the whole span of the blade
    blade_internal_structure = update_internal_structure(
        blade_internal_structure, blade_outer_shape_bem
    )

    blade.leband = _add_le_bands(blade_internal_structure["layers"])
    blade.teband = _add_te_bands(blade_internal_structure["layers"])

    ### COMPONENTS
    _add_components(blade, blade_internal_structure)

    _add_spar_caps(blade, blade_internal_structure["layers"])

    blade.updateBlade()
    # save(blade_name)
    # BladeDef_to_NuMADfile(obj,numad_name,matdb_name,numad_af_folder)
    return blade


def _add_stations(
    blade, blade_outer_shape_bem, hub_outer_shape_bem, af_data, file: str, write_airfoils
):
    # Obtaining some parameters not explicitly given in YAML file
    L = np.ceil(blade_outer_shape_bem["reference_axis"]["z"]["values"][-1])
    R = L + hub_outer_shape_bem["diameter"] / 2
    L = R - hub_outer_shape_bem["diameter"] / 2
    blade.span = np.multiply(np.transpose(blade_outer_shape_bem["reference_axis"]["z"]["grid"]), L)
    blade.ispan = blade.span

    # Aerodynamic properties
    # using interp because yaml can have different r/R for twist and chord
    temp_x = np.transpose(blade_outer_shape_bem["twist"]["grid"])
    temp_y = blade_outer_shape_bem["twist"]["values"]
    blade.degreestwist = (
        interpolator_wrap(np.multiply(temp_x, L), np.transpose(temp_y), blade.span) * 180.0 / np.pi
    )
    blade.chord = interpolator_wrap(
        np.multiply(np.transpose(blade_outer_shape_bem["chord"]["grid"]), L),
        np.transpose(blade_outer_shape_bem["chord"]["values"]),
        blade.span,
    )
    af_dir_names = []
    for i in range(len(af_data)):
        af_dir_names.append(af_data[i]["name"])
    numstations = len(blade_outer_shape_bem["airfoil_position"]["labels"])
    tc = [None] * numstations
    aero_cent = [None] * numstations

    for i in range(numstations):
        _, _, iaf_temp = np.intersect1d(
            blade_outer_shape_bem["airfoil_position"]["labels"][i],
            af_dir_names,
            "stable",
            return_indices=True,
        )
        IAF = iaf_temp[0]  # Expect only one index of intersection
        tc[i] = af_data[IAF]["relative_thickness"]
        tc_xL = blade_outer_shape_bem["airfoil_position"]["grid"][i]
        aero_cent[i] = af_data[IAF]["aerodynamic_center"]
        xf_coords = np.stack(
            (af_data[IAF]["coordinates"]["x"], af_data[IAF]["coordinates"]["y"]), 1
        )

        # find coordinate direction (clockwise or counter-clockwise) Winding
        # Number. clockwise starting at (1,0) is correct
        with np.errstate(divide="ignore", invalid="ignore"):
            if np.nanmean(np.gradient(np.arctan(xf_coords[:, 1] / xf_coords[:, 0]))) > 0:
                xf_coords = np.flipud(xf_coords)

        if write_airfoils:
            import os

            out_folder = "yaml2BladeDef_" + file.replace(".yaml", "")
            # blade_name = out_folder + '/' + file.replace('.yaml','') + '_blade.mat'
            # matdb_name =...
            # numade_name =...

            # Creating folders
            os.makedirs(out_folder + "/af_coords/", exist_ok=True)
            # os.makedirs(out_folder+'/af_polars/', exist_ok = True)
            os.makedirs(out_folder + "/airfoil/", exist_ok=True)
            writeNuMADAirfoil(
                xf_coords,
                blade_outer_shape_bem["airfoil_position"]["labels"][i],
                out_folder
                + "/af_coords/"
                + blade_outer_shape_bem["airfoil_position"]["labels"][i]
                + ".txt",
            )

        ref = blade_outer_shape_bem["airfoil_position"]["labels"][i]
        af = Airfoil(coords=xf_coords, ref=ref)
        af.resample(spacing="half-cosine")
        blade.addStation(af, tc_xL * L)
    # Obtain some key blade attributes
    blade.percentthick = np.multiply(
        interpolator_wrap(
            np.multiply(blade_outer_shape_bem["airfoil_position"]["grid"], L), tc, blade.ispan
        ),
        100,
    )
    blade.aerocenter = interpolator_wrap(
        np.multiply(blade_outer_shape_bem["airfoil_position"]["grid"], L), aero_cent, blade.span
    )
    blade.chordoffset = interpolator_wrap(
        np.multiply(np.transpose(blade_outer_shape_bem["pitch_axis"]["grid"]), L),
        np.transpose(blade_outer_shape_bem["pitch_axis"]["values"]),
        blade.span,
    )
    blade.naturaloffset = 0
    blade.prebend = interpolator_wrap(
        np.multiply(np.transpose(blade_outer_shape_bem["reference_axis"]["x"]["grid"]), L),
        np.transpose(blade_outer_shape_bem["reference_axis"]["x"]["values"]),
        blade.span,
    )
    blade.sweep = interpolator_wrap(
        np.multiply(np.transpose(blade_outer_shape_bem["reference_axis"]["y"]["grid"]), L),
        np.transpose(blade_outer_shape_bem["reference_axis"]["y"]["values"]),
        blade.span,
    )


def _add_materials(blade, material_data):
    materials_dict = {}
    for i in range(len(material_data)):
        cur_mat = Material()
        cur_mat.name = material_data[i]["name"]
        if material_data[i]["orth"] == 1:
            cur_mat.type = "orthotropic"
        else:
            cur_mat.type = "isotropic"
        # Add ply thickness option if ply thickness exists in yaml
        try:
            cur_mat.layerthickness = material_data[i]["ply_t"] * 1000
        except KeyError:
            print(
                "Warning! material ply thickness "
                + material_data[i]["name"]
                + " not defined, assuming 1 mm thickness"
            )
            cur_mat.layerthickness = 1

        finally:
            pass

        # first
        cur_mat.uts = _parse_data(material_data[i].get("Xt", 0))
        cur_mat.ucs = -_parse_data(material_data[i].get("Xc", 0))
        cur_mat.uss = _parse_data(material_data[i].get("S", 0))
        cur_mat.xzit = 0.3
        cur_mat.xzic = 0.25
        cur_mat.yzit = 0.3
        cur_mat.yzic = 0.25
        try:
            cur_mat.g1g2 = material_data[i].get("GIc", 0) / material_data[i].get("GIIc", 0)
        except ZeroDivisionError:
            cur_mat.g1g2 = np.nan
        if "alp0" in material_data[i]:
            cur_mat.alp0 = _parse_data(material_data[i]["alp0"])
        else:
            cur_mat.alp0 = None
            cur_mat.etat = None
        try:
            # test if property is a list
            material_data[i]["E"] + []
        except TypeError:
            cur_mat.ex = _parse_data(material_data[i]["E"])
            cur_mat.ey = _parse_data(material_data[i]["E"])
            cur_mat.ez = _parse_data(material_data[i]["E"])
            cur_mat.gxy = _parse_data(material_data[i]["G"])
            cur_mat.gxz = _parse_data(material_data[i]["G"])
            cur_mat.gyz = _parse_data(material_data[i]["G"])
            cur_mat.prxy = _parse_data(material_data[i]["nu"])
            cur_mat.prxz = _parse_data(material_data[i]["nu"])
            cur_mat.pryz = _parse_data(material_data[i]["nu"])
        else:
            cur_mat.ex = _parse_data(material_data[i]["E"][0])
            cur_mat.ey = _parse_data(material_data[i]["E"][1])
            cur_mat.ez = _parse_data(material_data[i]["E"][2])
            cur_mat.gxy = _parse_data(material_data[i]["G"][0])
            cur_mat.gxz = _parse_data(material_data[i]["G"][1])
            cur_mat.gyz = _parse_data(material_data[i]["G"][2])
            cur_mat.prxy = _parse_data(material_data[i]["nu"][0])
            cur_mat.prxz = _parse_data(material_data[i]["nu"][1])
            cur_mat.pryz = _parse_data(material_data[i]["nu"][2])
        try:
            cur_mat.m = material_data[i]["m"]
        except KeyError:
            print(f"No fatigue exponent found for material: {material_data[i]['name']}")
        cur_mat.density = material_data[i]["rho"]
        # cur_mat.dens = mat_data[i]['rho']
        cur_mat.drydensity = material_data[i]["rho"]
        if "description" in material_data[i].keys() and "source" in material_data[i].keys():
            desc_sourc = [material_data[i]["description"], ", ", material_data[i]["source"]]
            cur_mat.reference = "".join(desc_sourc)
        else:
            cur_mat.reference = []

        materials_dict[cur_mat.name] = cur_mat
    blade.materials = materials_dict
    return


def _add_components(blade, blade_internal_structure):
    N_layer_comp = len(blade_internal_structure["layers"])
    component_list = list()
    for i in range(N_layer_comp):
        i_component_data = blade_internal_structure["layers"][i]
        cur_comp = Component()
        cur_comp.group = 0
        cur_comp.name = i_component_data["name"]
        #   comp['material'] = blade_internal_structure['layers']{i}['material'];
        # mat_names = [mat.name for mat in blade.materials]
        # C,IA,IB = np.intersect1d(mat_names,i_component_data['material'],return_indices=True)
        cur_comp.materialid = i_component_data["material"]
        try:
            cur_comp.fabricangle = np.mean(i_component_data["fiber_orientation"]["values"])
        finally:
            pass
        if "spar" in i_component_data["name"].lower():
            cur_comp.imethod = "pchip"
        else:
            cur_comp.imethod = "linear"
        # cur_comp.cp[:,0] = np.transpose(i_component_data['thickness']['grid'])
        cptemp1 = np.transpose(i_component_data["thickness"]["grid"])
        temp_n_layer = (
            np.multiply(np.transpose(i_component_data["thickness"]["values"]), 1000.0)
            / blade.materials[cur_comp.materialid].layerthickness
        )
        I_round_up = np.flatnonzero((temp_n_layer > 0.05) & (temp_n_layer < 0.5))
        cptemp2 = np.round(
            np.multiply(np.transpose(i_component_data["thickness"]["values"]), 1000.0)
            / blade.materials[cur_comp.materialid].layerthickness
        )
        cur_comp.cp = np.stack((cptemp1, cptemp2), axis=1)
        # if I_round_up.size > 0:
        #     cur_comp.cp[I_round_up,1] = 1 # increase n_layers from 0 to 1 for 0.05<n_layers<0.5
        #     comp['cp'](:,2) = cell2mat(blade_internal_structure['layers']{i}['thickness']['values'])'.*1000;  # use when each material ply is 1 mm
        cur_comp.pinnedends = 0
        component_list.append(cur_comp)

    for comp in range(len(component_list)):
        # uv coating
        if "uv" in blade_internal_structure["layers"][comp]["name"].lower():
            component_list[comp].hpextents = ["le", "te"]
            component_list[comp].lpextents = ["le", "te"]
            component_list[comp].cp[:, 1] = component_list[comp].cp[:, 1]

        # Shell skin1
        if "shell_skin_outer" in blade_internal_structure["layers"][comp]["name"].lower():
            component_list[comp].hpextents = ["le", "te"]
            component_list[comp].lpextents = ["le", "te"]
            # CK Change me when yaml is fixed!!!!
            component_list[comp].cp[:, 1] = component_list[comp].cp[:, 1]

        # LE Band
        if "le_reinf" in blade_internal_structure["layers"][comp]["name"].lower():
            component_list[comp].hpextents = ["le", "a"]
            component_list[comp].lpextents = ["le", "a"]

        # TE Band
        if "te_reinf" in blade_internal_structure["layers"][comp]["name"].lower():
            component_list[comp].hpextents = ["d", "te"]
            component_list[comp].lpextents = ["d", "te"]

        # Trailing edge suction surface panel
        if "te_ss" in blade_internal_structure["layers"][comp]["name"].lower():
            component_list[comp].lpextents = ["c", "d"]

        # Leading edge suction surface panel
        if "le_ss" in blade_internal_structure["layers"][comp]["name"].lower():
            component_list[comp].lpextents = ["a", "b"]

        # Leading edge pressure surface panel)
        if "le_ps" in blade_internal_structure["layers"][comp]["name"].lower():
            component_list[comp].hpextents = ["a", "b"]

        # Trailing edge pressure surface panel
        if "te_ps" in blade_internal_structure["layers"][comp]["name"].lower():
            component_list[comp].hpextents = ["c", "d"]

        # Shell skin2
        if "shell_skin_inner" in blade_internal_structure["layers"][comp]["name"].lower():
            component_list[comp].hpextents = np.array(["le", "te"])
            component_list[comp].lpextents = np.array(["le", "te"])
            # CK Change me when yaml is fixed!!!!
            component_list[comp].cp[:, 1] = component_list[comp].cp[:, 1]

        # Forward Shear
        if "web" in blade_internal_structure["layers"][comp]:
            if (
                "fore" in blade_internal_structure["layers"][comp]["web"].lower()
                or "1" in blade_internal_structure["layers"][comp]["name"]
            ):
                # Web Skin1
                if "skinle" in blade_internal_structure["layers"][comp]["name"].lower().replace(
                    "_", ""
                ):
                    # comp[comp].hpextents = {[num2str(xs,2) 'b-c']};
                    # comp[comp].lpextents = {[num2str(xs,2) 'b-c']};
                    # comp[comp].hpextents = {['z+' sw_offset]};
                    # comp[comp].lpextents = {['z+' sw_offset]};
                    component_list[comp].hpextents = ["b"]
                    component_list[comp].lpextents = ["b"]
                    component_list[comp].group = 1
                    component_list[comp].name = blade_internal_structure["layers"][comp]["name"]
                    # CK Change me when yaml is fixed!!!!
                    component_list[comp].cp[:, 1] = component_list[comp].cp[:, 1]

                # Web Filler
                if "filler" in blade_internal_structure["layers"][comp]["name"].lower():
                    # comp[comp].hpextents = {[num2str(xs,2) 'b-c']};
                    # comp[comp].lpextents = {[num2str(xs,2) 'b-c']};
                    # comp[comp].hpextents = {['z+' sw_offset]};
                    # comp[comp].lpextents = {['z+' sw_offset]};
                    component_list[comp].hpextents = ["b"]
                    component_list[comp].lpextents = ["b"]
                    component_list[comp].group = 1
                    component_list[comp].name = blade_internal_structure["layers"][comp]["name"]

                # Web Skin2
                if "skinte" in blade_internal_structure["layers"][comp]["name"].lower().replace(
                    "_", ""
                ):
                    # comp[comp].hpextents = {[num2str(xs,2) 'b-c']};
                    # comp[comp].lpextents = {[num2str(xs,2) 'b-c']};
                    # comp[comp].hpextents = {['z+' sw_offset]};
                    # comp[comp].lpextents = {['z+' sw_offset]};
                    component_list[comp].hpextents = ["b"]
                    component_list[comp].lpextents = ["b"]
                    component_list[comp].group = 1
                    component_list[comp].name = blade_internal_structure["layers"][comp]["name"]
                    # CK Change me when yaml is fixed!!!!
                    component_list[comp].cp[:, 1] = component_list[comp].cp[:, 1]

        # Rear Shear
        if "web" in blade_internal_structure["layers"][comp]:
            if (
                "rear" in blade_internal_structure["layers"][comp]["web"].lower()
                or "0" in blade_internal_structure["layers"][comp]["name"]
            ):
                # Web Skin1
                if "skinle" in blade_internal_structure["layers"][comp]["name"].lower().replace(
                    "_", ""
                ):
                    # comp[comp].hpextents = {[num2str(xs,2) 'b-c']};
                    # comp[comp].lpextents = {[num2str(xs,2) 'b-c']};
                    # comp[comp].hpextents = {['z-' sw_offset]};
                    # comp[comp].lpextents = {['z-' sw_offset]};
                    component_list[comp].hpextents = ["c"]
                    component_list[comp].lpextents = ["c"]
                    component_list[comp].group = 2
                    component_list[comp].name = blade_internal_structure["layers"][comp]["name"]
                    # CK Change me when yaml is fixed!!!!
                    component_list[comp].cp[:, 1] = component_list[comp].cp[:, 1]
                # Web Filler
                if "filler" in blade_internal_structure["layers"][comp]["name"].lower():
                    # comp[comp].hpextents = {[num2str(xs,2) 'b-c']};
                    # comp[comp].lpextents = {[num2str(xs,2) 'b-c']};
                    # comp[comp].hpextents = {['z-' sw_offset]};
                    # comp[comp].lpextents = {['z-' sw_offset]};
                    component_list[comp].hpextents = ["c"]
                    component_list[comp].lpextents = ["c"]
                    component_list[comp].group = 2
                    component_list[comp].name = blade_internal_structure["layers"][comp]["name"]

                # Web Skin2
                if "skinte" in blade_internal_structure["layers"][comp]["name"].lower().replace(
                    "_", ""
                ):
                    # comp[comp].hpextents = {[num2str(xs,2) 'b-c']};
                    # comp[comp].lpextents = {[num2str(xs,2) 'b-c']};
                    # comp[comp].hpextents = {['z-' sw_offset]};
                    # comp[comp].lpextents = {['z-' sw_offset]};
                    component_list[comp].hpextents = ["c"]
                    component_list[comp].lpextents = ["c"]
                    component_list[comp].group = 2
                    component_list[comp].name = blade_internal_structure["layers"][comp]["name"]
                    # CK Change me when yaml is fixed!!!!
                    component_list[comp].cp[:, 1] = component_list[comp].cp[:, 1]

    ### add components to blade
    component_dict = {}
    for comp in component_list:
        component_dict[comp.name] = comp
    blade.components = component_dict


def writeNuMADAirfoil(coords, reftext, fname):
    """
    WriteNuMADAirfoil  Write NuMAD airfoil files
    **********************************************************************
    *                   Part of the SNL NuMAD Toolbox                    *
    * Developed by Sandia National Laboratories Wind Energy Technologies *
    *             See license.txt for disclaimer information             *
    **********************************************************************
      WriteNuMADAirfoil(coords,reftext,fname)

            fname - full filename, incl extension, of NuMAD airfoil file to write
        coords - Nx2 array of airfoil coordinate data.  First column contains
        x-values, second column contains y-values.  Airfoil coordinates are in
        order as specified by NuMAD (i.e. trailing edge = (1,0) and leading
        edge = (0,0)
        reftext = string representing reference text
    """
    with open(fname, "wt") as fid:
        fid.write("<reference>\n%s</reference>\n" % (reftext))
        fid.write("<coords>\n" % ())
        for i in range(coords.shape[0]):
            fid.write("%8.12f\t%8.12f\n" % tuple(coords[i, :]))

        fid.write("</coords>" % ())


def update_internal_structure(blade_internal_structure, blade_outer_shape_bem):
    bladeParts = ["layers", "webs"]
    # Make sure each blade.ispan has layer thicknesses and widths
    fullSpanGrid = np.array(blade_outer_shape_bem["reference_axis"]["z"]["grid"])
    nStations = len(fullSpanGrid)
    keysToModify = {
        "offset_y_pa",
        "thickness",
        "fiber_orientation",
        "width",
        "start_nd_arc",
        "end_nd_arc",
    }
    for part_name in bladeParts:
        N_layer_comp = len(blade_internal_structure[part_name])
        for currentLayer in range(N_layer_comp):
            layerKeys = set(blade_internal_structure[part_name][currentLayer].keys())

            for currentKey in keysToModify.intersection(layerKeys):
                try:
                    grid = blade_internal_structure[part_name][currentLayer][currentKey]["grid"]
                    values = blade_internal_structure[part_name][currentLayer][currentKey]["values"]
                except KeyError:
                    continue

                startStationLoc = grid[0]
                endStationLoc = grid[-1]

                subSpanGridIndex = np.where(
                    (fullSpanGrid >= startStationLoc) & (fullSpanGrid <= endStationLoc)
                )[0]

                # iterpolate fullSpanGrid locations onto layer grid defined in the yamle file for the layer
                subSpanValues = interpolator_wrap(
                    grid, values, fullSpanGrid[subSpanGridIndex], "pchip"
                )
                fullSpanValues = np.zeros(nStations)

                fullSpanValues[subSpanGridIndex] = subSpanValues

                # Reset
                blade_internal_structure[part_name][currentLayer][currentKey]["grid"] = fullSpanGrid
                blade_internal_structure[part_name][currentLayer][currentKey]["values"] = (
                    fullSpanValues
                )
    return blade_internal_structure


def _add_te_bands(blade_structure_dict):
    teReinfKeys = [
        l
        for l in blade_structure_dict
        if "te" in l["name"].lower() and "reinf" in l["name"].lower()
    ]
    if len(teReinfKeys) == 1:
        teband = teReinfKeys[0]["width"]["values"] * 1000 / 2
    elif len(teReinfKeys) == 2:
        teband = (teReinfKeys[0]["width"]["values"] + teReinfKeys[1]["width"]["values"]) * 1000 / 2
    else:
        raise ValueError("Unknown number of TE reinforcements")
    return teband


def _add_le_bands(blade_structure_dict):
    leReinfKeys = [
        l
        for l in blade_structure_dict
        if "le" in l["name"].lower() and "reinf" in l["name"].lower()
    ]
    if len(leReinfKeys) == 1:
        leband = leReinfKeys[0]["width"]["values"] * 1000 / 2
    elif len(leReinfKeys) == 2:
        leband = (leReinfKeys[0]["width"]["values"] + leReinfKeys[1]["width"]["values"]) * 1000 / 2
    else:
        raise ValueError("Invalid number of LE reinforcements")
    return leband


def _add_spar_caps(blade, blade_structure_dict):
    sparCaps = [cap for cap in blade_structure_dict if "spar" in cap["name"].lower()]

    if len(sparCaps) != 2:
        raise ValueError("Incorrect number of spar cap components")

    for iSparCap in range(2):
        if "suc" in sparCaps[iSparCap]["side"].lower():
            lpSideIndex = iSparCap
        if "pres" in sparCaps[iSparCap]["side"].lower():
            hpSideIndex = iSparCap

    blade.sparcapwidth_lp = sparCaps[lpSideIndex]["width"]["values"] * 1000
    try:
        blade.sparcapoffset_lp = sparCaps[lpSideIndex]["offset_y_pa"]["values"] * 1000
    except KeyError:
        blade.sparcap_start_nd_arc = sparCaps[lpSideIndex]["start_nd_arc"]["values"]
        blade.sparcap_end_nd_arc = sparCaps[lpSideIndex]["end_nd_arc"]["values"]

    blade.sparcapwidth_hp = sparCaps[hpSideIndex]["width"]["values"] * 1000
    try:
        blade.sparcapoffset_hp = sparCaps[hpSideIndex]["offset_y_pa"]["values"] * 1000
    except KeyError:
        blade.sparcap_start_nd_arc = sparCaps[hpSideIndex]["start_nd_arc"]["values"]
        blade.sparcap_end_nd_arc = sparCaps[hpSideIndex]["end_nd_arc"]["values"]
    # Spar Caps (pressure and suction)
    blade.components[sparCaps[hpSideIndex]["name"]].hpextents = ["b", "c"]
    blade.components[sparCaps[lpSideIndex]["name"]].lpextents = ["b", "c"]
