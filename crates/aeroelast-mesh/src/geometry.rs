/// Geometry utilities for mesh elements.
///
/// Includes vectorized span-direction angle offset computation
/// (replaces Python `_batch_angle_offsets`).

use rayon::prelude::*;

/// Compute span-direction angle offsets [degrees] for a batch of elements.
///
/// For each element, computes the local element orthonormal basis (e1, e2, e3)
/// and returns the angle between the span direction and the element's e1 axis,
/// measured in the element tangent plane.
///
/// # Arguments
/// * `node_coords` — flat (n_nodes × 3) array: `[x0,y0,z0, x1,y1,z1, ...]`
/// * `connectivity` — per-element 0-based node indices (only first 4 used for QUAD)
/// * `span_dir`     — 3-component span direction unit vector
///
/// # Returns
/// `Vec<f64>` of length `n_elements`, angle in degrees for each element.
/// Elements where `span_dir` is parallel to the normal get angle = 0.
pub fn batch_angle_offsets(
    node_coords: &[f64],
    connectivity: &[Vec<usize>],
    span_dir: [f64; 3],
) -> Vec<f64> {
    connectivity
        .par_iter()
        .map(|conn| element_angle_offset(node_coords, conn, span_dir))
        .collect()
}

/// Compute angle offset for a single element.
///
/// Returns the angle (degrees) between the projected span direction
/// and the element's local e1 axis.
pub fn element_angle_offset(
    node_coords: &[f64],
    conn: &[usize],
    span_dir: [f64; 3],
) -> f64 {
    if conn.len() < 3 {
        return 0.0;
    }

    let pt = |i: usize| -> [f64; 3] {
        let base = conn[i] * 3;
        [node_coords[base], node_coords[base + 1], node_coords[base + 2]]
    };

    let p0 = pt(0);
    let p1 = pt(1);
    let p2 = pt(2);

    let v1 = sub3(p1, p0);
    let v2 = sub3(p2, p0);

    // Average element normal
    let n1 = cross3(v1, v2);
    let n1_norm = norm3(n1);

    let e3 = if conn.len() >= 4 {
        let p3 = pt(3);
        let v1b = sub3(p2, p0);
        let v2b = sub3(p3, p0);
        let n2 = cross3(v1b, v2b);
        let n2_norm = norm3(n2);

        match (n1_norm > 1e-12, n2_norm > 1e-12) {
            (true, true) => {
                let avg = add3(scale3(n1, 1.0 / n1_norm), scale3(n2, 1.0 / n2_norm));
                let len = norm3(avg);
                if len > 1e-30 { scale3(avg, 1.0 / len) } else { [0.0, 0.0, 1.0] }
            }
            (true, false) => scale3(n1, 1.0 / n1_norm),
            (false, true) => scale3(n2, 1.0 / n2_norm),
            (false, false) => [0.0, 0.0, 1.0],
        }
    } else {
        if n1_norm > 1e-12 { scale3(n1, 1.0 / n1_norm) } else { [0.0, 0.0, 1.0] }
    };

    // e1 = edge 01 projected onto plane perpendicular to e3
    let dot_v1_e3 = dot3(v1, e3);
    let mut e1 = sub3(v1, scale3(e3, dot_v1_e3));
    let mut e1_norm = norm3(e1);

    if e1_norm < 1e-12 {
        // Fallback: use edge 02
        let dot_v2_e3 = dot3(v2, e3);
        e1 = sub3(v2, scale3(e3, dot_v2_e3));
        e1_norm = norm3(e1);
    }

    if e1_norm < 1e-30 {
        return 0.0;
    }
    e1 = scale3(e1, 1.0 / e1_norm);

    let e2_raw = cross3(e3, e1);
    let e2_norm = norm3(e2_raw);
    if e2_norm < 1e-30 {
        return 0.0;
    }
    let e2 = scale3(e2_raw, 1.0 / e2_norm);

    // Project span_dir onto element plane
    let dot_sd_e3 = dot3(span_dir, e3);
    let sd_proj = sub3(span_dir, scale3(e3, dot_sd_e3));
    let sd_norm = norm3(sd_proj);

    if sd_norm < 1e-10 {
        return 0.0;
    }

    let cos_a = dot3(sd_proj, e1) / sd_norm;
    let sin_a = dot3(sd_proj, e2) / sd_norm;

    sin_a.atan2(cos_a).to_degrees()
}

// --- tiny inline vector math ---

#[inline(always)]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline(always)]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline(always)]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline(always)]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
fn norm3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

#[inline(always)]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
