/// Mesh topology for shell FEM assembly.
///
/// Stores node coordinates, element connectivity, and DOF mapping.
/// Pure data structure — no PETSc dependency.

/// Element type for shell elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemType {
    /// MITC3+ triangular shell element (3 nodes × 6 DOFs = 18 DOFs)
    Mitc3,
    /// MITC4+ quadrilateral shell element (4 nodes × 6 DOFs = 24 DOFs)
    Mitc4,
}

/// Mesh topology for a collection of shell elements.
///
/// All indices are 0-based. Node coordinates are stored flat as
/// [x0,y0,z0, x1,y1,z1, ...] of length `n_nodes * 3`.
pub struct MeshTopology {
    /// Node coordinates (flat): length = n_nodes * 3
    pub node_coords: Vec<f64>,
    /// Element connectivity: `connectivity[e]` = slice of node indices (0-based)
    /// For Mitc3: 3 nodes; for Mitc4: 4 nodes.
    pub connectivity: Vec<Vec<usize>>,
    /// Element type for each element
    pub elem_types: Vec<ElemType>,
    /// Number of nodes
    pub n_nodes: usize,
    /// Number of elements
    pub n_elems: usize,
    /// DOFs per node (always 6 for shell elements: u,v,w,θx,θy,θz)
    pub dofs_per_node: usize,
}

impl MeshTopology {
    /// Create a new MeshTopology.
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
        MeshTopology {
            node_coords,
            connectivity,
            elem_types,
            n_nodes,
            n_elems,
            dofs_per_node: 6,
        }
    }

    /// Total number of DOFs in the system.
    pub fn dofs_count(&self) -> usize {
        self.n_nodes * self.dofs_per_node
    }

    /// Global DOF indices for element `elem_idx`.
    ///
    /// For each node in the element's connectivity, returns 6 consecutive DOF
    /// indices: [6*node, 6*node+1, ..., 6*node+5].
    /// The result has length `n_nodes_per_elem * 6`.
    pub fn global_dof_indices(&self, elem_idx: usize) -> Vec<usize> {
        let conn = &self.connectivity[elem_idx];
        let mut dofs = Vec::with_capacity(conn.len() * self.dofs_per_node);
        for &node in conn {
            for d in 0..self.dofs_per_node {
                dofs.push(node * self.dofs_per_node + d);
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
    /// For Mitc3: 9 values; for Mitc4: 12 values.
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
