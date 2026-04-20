/// HDF5 mesh reader.
///
/// Reads the format written by Python `write_hdf5` / `load_hdf5`:
///
/// ```text
/// /nodes/coords            f64[n_nodes, 3]
/// /nodes/ids               i64[n_nodes]
/// /nodes/geometric_node    u8[n_nodes]   (1 = geometric, 0 = midside)
/// /elements/ids            i64[n_elems]
/// /elements/types          i64[n_elems]  (VTK cell type codes)
/// /elements/connectivity   i64[total_nodes]
/// /elements/connectivity_offsets i64[n_elems+1]
/// /node_sets/<name>/node_ids     i64[n]
/// /element_sets/<name>/element_ids i64[n]
/// ```
use hdf5::File;
use ndarray::Array2;

use crate::entities::{Element, ElementSet, ElementType, Node, NodeSet};
use crate::error::MeshError;
use crate::model::MeshModel;

/// Load a `MeshModel` from an HDF5 file written by the Python writer.
pub fn load_hdf5(path: &str) -> Result<MeshModel, MeshError> {
    let file = File::open(path).map_err(|e| MeshError::Hdf5(e.to_string()))?;

    let mut mesh = MeshModel::new();

    // ---- Nodes ----
    let nodes_grp = file
        .group("nodes")
        .map_err(|e| MeshError::Hdf5(e.to_string()))?;

    let coords: Array2<f64> = nodes_grp
        .dataset("coords")
        .and_then(|d| d.read_2d::<f64>())
        .map_err(|e| MeshError::Hdf5(e.to_string()))?;

    let node_ids: Vec<i64> = nodes_grp
        .dataset("ids")
        .and_then(|d| d.read_1d::<i64>())
        .map(|a| a.to_vec())
        .map_err(|e| MeshError::Hdf5(e.to_string()))?;

    let n_nodes = node_ids.len();
    for i in 0..n_nodes {
        let node = Node::new(
            node_ids[i] as u64,
            coords[[i, 0]],
            coords[[i, 1]],
            coords[[i, 2]],
        );
        mesh.add_node(node)?;
    }

    // ---- Elements ----
    let elems_grp = file
        .group("elements")
        .map_err(|e| MeshError::Hdf5(e.to_string()))?;

    let element_ids: Vec<i64> = elems_grp
        .dataset("ids")
        .and_then(|d| d.read_1d::<i64>())
        .map(|a| a.to_vec())
        .map_err(|e| MeshError::Hdf5(e.to_string()))?;

    let element_types: Vec<i64> = elems_grp
        .dataset("types")
        .and_then(|d| d.read_1d::<i64>())
        .map(|a| a.to_vec())
        .map_err(|e| MeshError::Hdf5(e.to_string()))?;

    let connectivity: Vec<i64> = elems_grp
        .dataset("connectivity")
        .and_then(|d| d.read_1d::<i64>())
        .map(|a| a.to_vec())
        .map_err(|e| MeshError::Hdf5(e.to_string()))?;

    let offsets: Vec<i64> = elems_grp
        .dataset("connectivity_offsets")
        .and_then(|d| d.read_1d::<i64>())
        .map(|a| a.to_vec())
        .map_err(|e| MeshError::Hdf5(e.to_string()))?;

    let n_elems = element_ids.len();
    for i in 0..n_elems {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        let node_ids_elem: Vec<u64> = connectivity[start..end]
            .iter()
            .map(|&nid| nid as u64)
            .collect();

        let etype = vtk_code_to_element_type(element_types[i]);
        let elem = Element::new(element_ids[i] as u64, node_ids_elem, etype);
        mesh.add_element(elem)?;
    }

    // ---- Node sets ----
    if let Ok(ns_grp) = file.group("node_sets") {
        for name in ns_grp.member_names().unwrap_or_default() {
            if let Ok(ids) = ns_grp
                .group(&name)
                .and_then(|g| g.dataset("node_ids"))
                .and_then(|d| d.read_1d::<i64>())
                .map(|a| a.to_vec())
            {
                let node_ids: Vec<u64> = ids.into_iter().map(|x| x as u64).collect();
                mesh.add_node_set(NodeSet::with_ids(name, node_ids));
            }
        }
    }

    // ---- Element sets ----
    if let Ok(es_grp) = file.group("element_sets") {
        for name in es_grp.member_names().unwrap_or_default() {
            if let Ok(ids) = es_grp
                .group(&name)
                .and_then(|g| g.dataset("element_ids"))
                .and_then(|d| d.read_1d::<i64>())
                .map(|a| a.to_vec())
            {
                let elem_ids: Vec<u64> = ids.into_iter().map(|x| x as u64).collect();
                mesh.add_element_set(ElementSet::with_ids(name, elem_ids));
            }
        }
    }

    Ok(mesh)
}

/// Map VTK cell type codes → `ElementType`.
///
/// Uses the VTK numeric codes that correspond to the Python `ElementType` IntEnum
/// (which inherits from `pyvista.CellType`).
fn vtk_code_to_element_type(code: i64) -> ElementType {
    match code {
        5  => ElementType::Triangle3,
        22 => ElementType::Triangle6,
        9  => ElementType::Quad4,
        23 => ElementType::Quad8,
        36 => ElementType::Quad9,
        10 => ElementType::Tetra4,
        24 => ElementType::Tetra10,
        12 => ElementType::Hexa8,
        25 => ElementType::Hexa20,
        13 => ElementType::Wedge6,
        26 => ElementType::Wedge15,
        14 => ElementType::Pyramid5,
        27 => ElementType::Pyramid13,
        _  => ElementType::Quad4, // fallback
    }
}
