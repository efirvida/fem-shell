#!/usr/bin/env python3
"""
Generate blockMeshDict for multi-flap FSI simulations.

Reads configuration from case_config.json in the case root directory.
"""

import json
from pathlib import Path


def main():
    base = Path(__file__).parent
    case_root = base.parent.parent  # fluid/system -> fluid -> case_root

    # Try to read from shared config first, fallback to local
    config_path = case_root / "case_config.json"
    if not config_path.exists():
        config_path = base / "blockMeshParams.json"

    out_path = base / "blockMeshDict"

    with config_path.open() as f:
        p = json.load(f)

    # Domain geometry
    inlet_dist = float(p["domain"]["inlet_distance"])
    outlet_dist = float(p["domain"]["outlet_distance"])
    domain_h = float(p["domain"]["domain_height"])

    # Flap configuration
    n_flaps = int(p["flaps"]["number"])
    flap_w = float(p["flaps"]["width"])
    flap_h = float(p["flaps"]["height"])
    x_spacing = float(p["flaps"]["x_spacing"])

    # Mesh parameters - support both old and new config format
    mesh_cfg = p.get("fluid_mesh", p.get("mesh", {}))
    z_thick = float(mesh_cfg["z_thickness"])

    inlet_nx = int(mesh_cfg["inlet_nx"])
    outlet_nx = int(mesh_cfg["outlet_nx"])
    gap_nx = int(mesh_cfg["gap_nx"])
    flap_nx = int(mesh_cfg["flap_nx"])

    # Boundary layer parameters
    first_cell_height = float(mesh_cfg["first_cell_height"])
    expansion_ratio = float(mesh_cfg["expansion_ratio"])

    # Grading (X only, Y grading is determined by expansion_ratio)
    grading_cfg = mesh_cfg.get("grading", p.get("grading", {}))
    grade_inlet_x = float(grading_cfg.get("inlet_x_grade", 1.0))
    grade_outlet_x = float(grading_cfg.get("outlet_x_grade", 1.0))

    # Z coords - centered at z=0 to avoid preCICE 2D warnings
    z0 = -z_thick / 2.0
    z1 = z_thick / 2.0

    # Build X coordinates: inlet -> flaps -> outlet
    # Total flap array width
    total_flap_width = n_flaps * flap_w + (n_flaps - 1) * x_spacing
    center_x = 0.0
    x_start = center_x - 0.5 * total_flap_width
    x_end = x_start + total_flap_width

    x0 = x_start - inlet_dist
    x3 = x_end + outlet_dist

    # Build X array: [x0, inlet_region, flap_regions, outlet_region, x3]
    xs = [x0]
    flap_pairs = []
    for i in range(n_flaps):
        xf_left = x_start + i * (flap_w + x_spacing)
        xf_right = xf_left + flap_w
        flap_pairs.append((xf_left, xf_right))
        xs.append(xf_left)
        xs.append(xf_right)
    xs.append(x3)
    xs = sorted(set(xs))  # Remove duplicates and sort

    # Build Y coordinates: ground, flap height, domain top
    ys = [0.0, flap_h, domain_h]

    # Determine which X intervals are flap thickness vs. gap
    flap_intervals = set(flap_pairs)

    def v_index(ix, iy, iz):
        nx = len(xs)
        ny = len(ys)
        idx2d = iy * nx + ix
        return idx2d + (0 if iz == 0 else ny * nx)

    lines = []
    lines.append("FoamFile")
    lines.append("{")
    lines.append("    version     2.0;")
    lines.append("    format      ascii;")
    lines.append("    class       dictionary;")
    lines.append("    object      blockMeshDict;")
    lines.append("}")
    lines.append("")

    vertices = []
    for iz, z in enumerate([z0, z1]):
        for iy, yv in enumerate(ys):
            for xv in xs:
                vertices.append((xv, yv, z))

    lines.append("vertices")
    lines.append("(")
    nx = len(xs)
    ny = len(ys)
    for i, (x, yv, z) in enumerate(vertices[: ny * nx]):
        lines.append(f"    ({x} {yv} {z} )         // {i}")
    for i, (x, yv, z) in enumerate(vertices[ny * nx :]):
        lines.append(f"    ({x} {yv} {z} )         // {i + ny * nx}")
    lines.append(");")
    lines.append("")

    lines.append("blocks")
    lines.append("(")

    # Calculate mesh parameters for bottom layer based on:
    # - first_cell_height: height of first cell at lowerWall
    # - expansion_ratio: ratio between consecutive cells (cell-to-cell)
    #
    # Geometric series: h1, h1*r, h1*r^2, ..., h1*r^(n-1)
    # Sum = h1 * (r^n - 1) / (r - 1) = flap_h
    #
    # Solve for n (number of cells):
    # r^n = 1 + (flap_h / h1) * (r - 1)
    # n = log(1 + (flap_h / h1) * (r - 1)) / log(r)

    import math

    h1 = first_cell_height
    r = expansion_ratio

    if abs(r - 1.0) < 1e-10:
        # Uniform cells
        ny_bottom = int(round(flap_h / h1))
        grade_vert = 1.0
        h_last = h1
    else:
        # Calculate number of cells needed
        n_float = math.log(1 + (flap_h / h1) * (r - 1)) / math.log(r)
        ny_bottom = max(2, int(round(n_float)))

        # Recalculate actual last cell height with integer n
        # Sum = h1 * (r^n - 1) / (r - 1)
        # We adjust h1 slightly to match exactly flap_h
        actual_sum = h1 * (r**ny_bottom - 1) / (r - 1)

        # simpleGrading = ratio of last cell to first cell = r^(n-1)
        grade_vert = r ** (ny_bottom - 1)

        # Last cell height
        h_last = h1 * (r ** (ny_bottom - 1))

    # Top layer: uniform cells with height = h_last (last cell of bottom layer)
    top_height = domain_h - flap_h
    ny_top = max(1, int(round(top_height / h_last)))

    # Pre-calculate nx for each X segment to ensure consistency across Y layers
    nx_per_segment = []
    for ix in range(len(xs) - 1):
        seg = (xs[ix], xs[ix + 1])

        # Determine segment type and apply corresponding nx
        if seg in flap_intervals:
            # Flap thickness segment
            nx_val = flap_nx
        elif xs[ix] == x0:
            # Inlet region (first segment)
            nx_val = inlet_nx
        elif xs[ix + 1] == x3:
            # Outlet region (last segment)
            nx_val = outlet_nx
        else:
            # Gap between flaps
            nx_val = gap_nx

        nx_per_segment.append(nx_val)

    # Bottom layer (y=0 to y=flap_h)
    for ix in range(len(xs) - 1):
        seg = (xs[ix], xs[ix + 1])
        if seg in flap_intervals:
            continue  # No fluid block within flap thickness

        nx_seg = nx_per_segment[ix]

        # Apply X grading: refine towards flaps at inlet, expand at outlet
        if xs[ix] == x0:
            # Inlet: refine towards first flap
            gx = grade_inlet_x
        elif xs[ix + 1] == x3:
            # Outlet: expand away from last flap
            gx = grade_outlet_x
        else:
            gx = 1.0  # Gap regions uniform

        a = v_index(ix, 0, 0)
        b = v_index(ix + 1, 0, 0)
        c = v_index(ix + 1, 1, 0)
        d = v_index(ix, 1, 0)
        e = v_index(ix, 0, 1)
        f = v_index(ix + 1, 0, 1)
        g = v_index(ix + 1, 1, 1)
        h = v_index(ix, 1, 1)

        lines.append(f"    hex (   {a:<3} {b:<3} {c:<3} {d:<3} {e:<3} {f:<3} {g:<3} {h:<3} )")
        lines.append(f"    ({nx_seg} {ny_bottom} 1)")
        # Apply vertical grading ONLY if this segment is adjacent to a flap on at least one side
        # Otherwise use uniform grading (gy=1.0) for pure gap regions
        is_adjacent_to_flap = False
        if ix > 0 and (xs[ix - 1], xs[ix]) in flap_intervals:
            is_adjacent_to_flap = True  # Left side is flap
        elif ix < len(xs) - 2 and (xs[ix + 1], xs[ix + 2]) in flap_intervals:
            is_adjacent_to_flap = True  # Right side is flap
        gy = grade_vert if is_adjacent_to_flap else 1.0
        lines.append(f"    simpleGrading ({gx} {gy} 1.0)\n")

    # Top layer (y=flap_h to y=domain_h)
    for ix in range(len(xs) - 1):
        nx_seg = nx_per_segment[ix]

        # Apply X grading: refine towards flaps at inlet, expand at outlet
        if xs[ix] == x0:
            # Inlet: refine towards first flap
            gx = grade_inlet_x
        elif xs[ix + 1] == x3:
            # Outlet: expand away from last flap
            gx = grade_outlet_x
        else:
            gx = 1.0  # Gap regions uniform

        a = v_index(ix, 1, 0)
        b = v_index(ix + 1, 1, 0)
        c = v_index(ix + 1, 2, 0)
        d = v_index(ix, 2, 0)
        e = v_index(ix, 1, 1)
        f = v_index(ix + 1, 1, 1)
        g = v_index(ix + 1, 2, 1)
        h = v_index(ix, 2, 1)

        lines.append(f"    hex (   {a:<3} {b:<3} {c:<3} {d:<3} {e:<3} {f:<3} {g:<3} {h:<3} )")
        lines.append(f"    ({nx_seg} {ny_top} 1)")
        # Top layer: always uniform vertical grading (gy=1.0) to maintain aspect ratio
        # from the bottom layer's last cell height
        gy = 1.0
        lines.append(f"    simpleGrading ({gx} {gy} 1.0)\n")
    lines.append(");")
    lines.append("")

    # Boundary patches
    lines.append("boundary")
    lines.append("(")

    # inlet
    lines.append("    inlet")
    lines.append("    {")
    lines.append("        type patch;")
    lines.append("        faces")
    lines.append("        (")
    for k in range(len(ys) - 1):
        a = v_index(0, k, 0)
        b = v_index(0, k + 1, 0)
        c = v_index(0, k + 1, 1)
        d = v_index(0, k, 1)
        lines.append(f"            (   {a:<3} {b:<3} {c:<3} {d:<3}  )")
    lines.append("        );")
    lines.append("    }")

    # outlet
    lines.append("    outlet")
    lines.append("    {")
    lines.append("        type patch;")
    lines.append("        faces")
    lines.append("        (")
    last_ix = len(xs) - 1
    for k in range(len(ys) - 1):
        a = v_index(last_ix, k, 0)
        b = v_index(last_ix, k + 1, 0)
        c = v_index(last_ix, k + 1, 1)
        d = v_index(last_ix, k, 1)
        lines.append(f"            (   {a:<3} {b:<3} {c:<3} {d:<3}  )")
    lines.append("        );")
    lines.append("    }")

    # flap patches: one per flap or combined
    separate_patches = p.get("separate_flap_patches", True)
    if not separate_patches:
        lines.append("    flap")
        lines.append("    {")
        lines.append("        type wall;")
        lines.append("        faces")
        lines.append("        (")
        for i, (xf_left, xf_right) in enumerate(flap_pairs, start=1):
            left_ix = xs.index(xf_left)
            right_ix = xs.index(xf_right)
            # Left vertical face (at x=xf_left, y in [0, flap_h])
            a = v_index(left_ix, 0, 0)
            b = v_index(left_ix, 1, 0)
            c = v_index(left_ix, 1, 1)
            d = v_index(left_ix, 0, 1)
            lines.append(f"            (   {a:<3} {b:<3} {c:<3} {d:<3}  )")
            # Top horizontal face (at y=flap_h, x in [xf_left, xf_right])
            a = v_index(left_ix, 1, 0)
            b = v_index(right_ix, 1, 0)
            c = v_index(right_ix, 1, 1)
            d = v_index(left_ix, 1, 1)
            lines.append(f"            (   {a:<3} {b:<3} {c:<3} {d:<3}  )")
            # Right vertical face (at x=xf_right, y in [0, flap_h])
            a = v_index(right_ix, 0, 0)
            b = v_index(right_ix, 1, 0)
            c = v_index(right_ix, 1, 1)
            d = v_index(right_ix, 0, 1)
            lines.append(f"            (   {a:<3} {b:<3} {c:<3} {d:<3}  )")
        lines.append("        );")
        lines.append("    }")
    else:
        # Separate patch per flap
        for i, (xf_left, xf_right) in enumerate(flap_pairs, start=1):
            lines.append(f"    flap{i}")
            lines.append("    {")
            lines.append("        type wall;")
            lines.append("        faces")
            lines.append("        (")
            left_ix = xs.index(xf_left)
            right_ix = xs.index(xf_right)
            # Left vertical
            a = v_index(left_ix, 0, 0)
            b = v_index(left_ix, 1, 0)
            c = v_index(left_ix, 1, 1)
            d = v_index(left_ix, 0, 1)
            lines.append(f"            (   {a:<3} {b:<3} {c:<3} {d:<3}  )")
            # Top horizontal
            a = v_index(left_ix, 1, 0)
            b = v_index(right_ix, 1, 0)
            c = v_index(right_ix, 1, 1)
            d = v_index(left_ix, 1, 1)
            lines.append(f"            (   {a:<3} {b:<3} {c:<3} {d:<3}  )")
            # Right vertical
            a = v_index(right_ix, 0, 0)
            b = v_index(right_ix, 1, 0)
            c = v_index(right_ix, 1, 1)
            d = v_index(right_ix, 0, 1)
            lines.append(f"            (   {a:<3} {b:<3} {c:<3} {d:<3}  )")
            lines.append("        );")
            lines.append("    }")

    # lowerWall (ground, y=0, excluding flap thickness intervals)
    lines.append("    lowerWall")
    lines.append("    {")
    lines.append("        type wall;")
    lines.append("        faces")
    lines.append("        (")
    for ix in range(len(xs) - 1):
        seg = (xs[ix], xs[ix + 1])
        if seg in flap_intervals:
            continue
        a = v_index(ix, 0, 0)
        b = v_index(ix + 1, 0, 0)
        c = v_index(ix + 1, 0, 1)
        d = v_index(ix, 0, 1)
        lines.append(f"            (   {a:<3} {b:<3} {c:<3} {d:<3}  )")
    lines.append("        );")
    lines.append("    }")

    # upperWall (top, y=domain_h) - patch type for far-field/atmospheric boundary
    lines.append("    upperWall")
    lines.append("    {")
    lines.append("        type patch;")
    lines.append("        faces")
    lines.append("        (")
    for ix in range(len(xs) - 1):
        a = v_index(ix, 2, 0)
        b = v_index(ix + 1, 2, 0)
        c = v_index(ix + 1, 2, 1)
        d = v_index(ix, 2, 1)
        lines.append(f"            (   {a:<3} {b:<3} {c:<3} {d:<3}  )")
    lines.append("        );")
    lines.append("    }")

    # frontAndBack empty patch
    lines.append("    frontAndBack")
    lines.append("    {")
    lines.append("        type empty;")
    lines.append("        faces")
    lines.append("        (")
    # frontAndBack faces for all blocks
    # bottom layer blocks
    for ix in range(len(xs) - 1):
        seg = (xs[ix], xs[ix + 1])
        if seg in flap_intervals:
            continue
        a = v_index(ix, 0, 0)
        b = v_index(ix + 1, 0, 0)
        c = v_index(ix + 1, 1, 0)
        d = v_index(ix, 1, 0)
        lines.append(f"            (   {a:<3} {b:<3} {c:<3} {d:<3}  )")
        a = v_index(ix, 0, 1)
        b = v_index(ix + 1, 0, 1)
        c = v_index(ix + 1, 1, 1)
        d = v_index(ix, 1, 1)
        lines.append(f"            (   {a:<3} {b:<3} {c:<3} {d:<3}  )")
    # top layer blocks
    for ix in range(len(xs) - 1):
        a = v_index(ix, 1, 0)
        b = v_index(ix + 1, 1, 0)
        c = v_index(ix + 1, 2, 0)
        d = v_index(ix, 2, 0)
        lines.append(f"            (   {a:<3} {b:<3} {c:<3} {d:<3}  )")
        a = v_index(ix, 1, 1)
        b = v_index(ix + 1, 1, 1)
        c = v_index(ix + 1, 2, 1)
        d = v_index(ix, 2, 1)
        lines.append(f"            (   {a:<3} {b:<3} {c:<3} {d:<3}  )")
    lines.append("        );")
    lines.append("    }")
    lines.append(");")

    with out_path.open("w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {out_path}")
    print(f"Domain: X=[{x0:.2f}, {x3:.2f}], Y=[0, {domain_h:.2f}], Z=[{z0}, {z1}]")
    print(f"Flaps: {n_flaps} x (w={flap_w}, h={flap_h}), spacing={x_spacing}")
    print(f"Mesh: first_cell={h1}, expansion_ratio={r}, bottom_ny={ny_bottom}, top_ny={ny_top}")
    print(f"Grading: Y_simpleGrading={grade_vert:.4f}, last_cell_height={h_last:.6f}")


if __name__ == "__main__":
    main()
