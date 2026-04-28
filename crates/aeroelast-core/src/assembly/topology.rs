/// Mesh topology for shell FEM assembly.
///
/// Stores node coordinates, element connectivity, and DOF mapping.
/// Pure data structure — no PETSc dependency.

/// Element type for all supported FEM element families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemType {
    // ---- Shell (6 DOFs/node) ----
    /// MITC3+ triangular shell element (3 nodes × 6 DOFs = 18 DOFs)
    Mitc3,
    /// MITC4+ quadrilateral shell element (4 nodes × 6 DOFs = 24 DOFs)
    Mitc4,
    /// MITC3+ composite shell element (3 nodes × 6 DOFs = 18 DOFs)
    Mitc3Composite,
    /// MITC4+ composite shell element (4 nodes × 6 DOFs = 24 DOFs)
    Mitc4Composite,
    // ---- Plane/QUAD (2 DOFs/node) ----
    /// 4-node bilinear quadrilateral plane element (4 nodes × 2 DOFs = 8 DOFs)
    Quad4,
    /// 8-node serendipity quadrilateral plane element (8 nodes × 2 DOFs = 16 DOFs)
    Quad8,
    /// 9-node Lagrange quadrilateral plane element (9 nodes × 2 DOFs = 18 DOFs)
    Quad9,
    // ---- 3D Solid (3 DOFs/node) ----
    /// 8-node hexahedral solid element (8 nodes × 3 DOFs = 24 DOFs)
    Hexa8,
    /// 20-node serendipity hexahedral solid element (20 nodes × 3 DOFs = 60 DOFs)
    Hexa20,
    /// 4-node tetrahedral solid element (4 nodes × 3 DOFs = 12 DOFs)
    Tetra4,
    /// 10-node quadratic tetrahedral solid element (10 nodes × 3 DOFs = 30 DOFs)
    Tetra10,
    /// 6-node wedge (triangular prism) solid element (6 nodes × 3 DOFs = 18 DOFs)
    Wedge6,
    /// 15-node quadratic wedge solid element (15 nodes × 3 DOFs = 45 DOFs)
    Wedge15,
    /// 5-node pyramid solid element (5 nodes × 3 DOFs = 15 DOFs)
    Pyramid5,
    /// 13-node quadratic pyramid solid element (13 nodes × 3 DOFs = 39 DOFs)
    Pyramid13,
}

impl ElemType {
    /// Number of DOFs per node for this element type.
    pub fn dofs_per_node(self) -> usize {
        match self {
            ElemType::Mitc3 | ElemType::Mitc4 | ElemType::Mitc3Composite | ElemType::Mitc4Composite => 6,
            ElemType::Quad4 | ElemType::Quad8 | ElemType::Quad9 => 2,
            ElemType::Hexa8 | ElemType::Hexa20 | ElemType::Tetra4 | ElemType::Tetra10
            | ElemType::Wedge6 | ElemType::Wedge15 | ElemType::Pyramid5 | ElemType::Pyramid13 => 3,
        }
    }
}

/// Mesh topology for a collection of FEM elements.
///
/// All indices are 0-based. Node coordinates are stored flat as
/// [x0,y0,z0, x1,y1,z1, ...] of length `n_nodes * 3`.
///
/// The `dofs_per_node` field controls the global DOF stride. All nodes use
/// the same stride so that DOF indices never alias. For a pure solid mesh
/// use 3; for a pure shell mesh use 6; for a pure plane mesh use 2.
#[derive(Clone)]
pub struct MeshTopology {
    /// Node coordinates (flat): length = n_nodes * 3
    pub node_coords: Vec<f64>,
    /// Element connectivity: `connectivity[e]` = slice of node indices (0-based)
    pub connectivity: Vec<Vec<usize>>,
    /// Element type for each element
    pub elem_types: Vec<ElemType>,
    /// Number of nodes
    pub n_nodes: usize,
    /// Number of elements
    pub n_elems: usize,
    /// Global DOF stride per node (6 for shell, 3 for solid, 2 for plane).
    pub dofs_per_node: usize,
}

impl MeshTopology {
    /// Create a new MeshTopology.
    ///
    /// The global DOF stride (`dofs_per_node`) is inferred from the maximum
    /// `ElemType::dofs_per_node()` value across all elements. For uniform
    /// meshes this is simply the element family DOF count.
    ///
    /// # Arguments
    /// * `node_coords` - Flat node coordinates [x0,y0,z0, x1,y1,z1, ...]
    /// * `connectivity` - Per-element node index lists (0-based)
    /// * `elem_types` - Element type per element
    pub fn new(
        node_coords: Vec<f64>,
        connectivity: Vec<Vec<usize>>,
        elem_types: Vec<ElemType>,
    ) -> Self {
        assert_eq!(
            connectivity.len(),
            elem_types.len(),
            "connectivity and elem_types must have the same length"
        );
        let n_nodes = node_coords.len() / 3;
        let n_elems = connectivity.len();
        let dofs_per_node = elem_types.iter().map(|e| e.dofs_per_node()).max().unwrap_or(6);
        MeshTopology {
            node_coords,
            connectivity,
            elem_types,
            n_nodes,
            n_elems,
            dofs_per_node,
        }
    }

    /// Total number of DOFs in the system.
    pub fn dofs_count(&self) -> usize {
        self.n_nodes * self.dofs_per_node
    }

    /// Global DOF indices for element `elem_idx`.
    ///
    /// For each node in the element's connectivity, returns `dofs_per_node`
    /// consecutive DOF indices. The element only uses as many DOFs as its type
    /// requires (`elem_type.dofs_per_node()`), which may be ≤ `self.dofs_per_node`.
    pub fn global_dof_indices(&self, elem_idx: usize) -> Vec<usize> {
        let conn = &self.connectivity[elem_idx];
        let elem_dofs = self.elem_types[elem_idx].dofs_per_node();
        let mut dofs = Vec::with_capacity(conn.len() * elem_dofs);
        for &node in conn {
            let base = node * self.dofs_per_node;
            for d in 0..elem_dofs {
                dofs.push(base + d);
            }
        }
        dofs
    }

    /// Get node coordinates as [x, y, z] for node index `i`.
    pub fn node_xyz(&self, i: usize) -> [f64; 3] {
        [
            self.node_coords[3 * i],
            self.node_coords[3 * i + 1],
            self.node_coords[3 * i + 2],
        ]
    }

    /// Get coordinates for element `elem_idx` as a flat vector.
    /// Returns `n_nodes_in_elem * 3` values (always 3D coords from the node table).
    pub fn elem_coords(&self, elem_idx: usize) -> Vec<f64> {
        let conn = &self.connectivity[elem_idx];
        let mut coords = Vec::with_capacity(conn.len() * 3);
        for &node in conn {
            coords.push(self.node_coords[3 * node]);
            coords.push(self.node_coords[3 * node + 1]);
            coords.push(self.node_coords[3 * node + 2]);
        }
        coords
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 2-element patch: 2 MITC3 triangles sharing edge (1,2).
    ///
    /// Nodes:
    ///   0: (0,0,0)
    ///   1: (1,0,0)
    ///   2: (0,1,0)
    ///   3: (1,1,0)
    ///
    /// Elements:
    ///   0: [0, 1, 2]
    ///   1: [1, 3, 2]
    fn two_tri_patch() -> MeshTopology {
        let node_coords = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
        ];
        let connectivity = vec![vec![0, 1, 2], vec![1, 3, 2]];
        let elem_types = vec![ElemType::Mitc3, ElemType::Mitc3];
        MeshTopology::new(node_coords, connectivity, elem_types)
    }

    #[test]
    fn test_dofs_count() {
        let topo = two_tri_patch();
        assert_eq!(topo.dofs_count(), 4 * 6); // 4 nodes × 6 DOFs
    }

    #[test]
    fn test_global_dof_indices_elem0() {
        let topo = two_tri_patch();
        let dofs = topo.global_dof_indices(0);
        // Element 0: nodes [0,1,2] → DOFs [0..5, 6..11, 12..17]
        assert_eq!(dofs.len(), 18);
        for i in 0..6 {
            assert_eq!(dofs[i], i);        // node 0
            assert_eq!(dofs[6 + i], 6 + i); // node 1
            assert_eq!(dofs[12 + i], 12 + i); // node 2
        }
    }

    #[test]
    fn test_global_dof_indices_elem1() {
        let topo = two_tri_patch();
        let dofs = topo.global_dof_indices(1);
        // Element 1: nodes [1,3,2] → DOFs [6..11, 18..23, 12..17]
        assert_eq!(dofs.len(), 18);
        for i in 0..6 {
            assert_eq!(dofs[i], 6 + i);    // node 1
            assert_eq!(dofs[6 + i], 18 + i); // node 3
            assert_eq!(dofs[12 + i], 12 + i); // node 2
        }
    }

    #[test]
    fn test_elem_coords() {
        let topo = two_tri_patch();
        let coords = topo.elem_coords(0);
        assert_eq!(coords.len(), 9);
        // Node 0: (0,0,0)
        assert_eq!(coords[0..3], [0.0, 0.0, 0.0]);
        // Node 1: (1,0,0)
        assert_eq!(coords[3..6], [1.0, 0.0, 0.0]);
        // Node 2: (0,1,0)
        assert_eq!(coords[6..9], [0.0, 1.0, 0.0]);
    }
}
