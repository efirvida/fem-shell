use std::collections::HashMap;

use crate::entities::{Element, ElementSet, Node, NodeSet};
use crate::error::MeshError;

/// Complete mesh model: nodes, elements, sets, and index lookup caches.
///
/// Mirrors the Python `MeshModel` but stores everything as flat Vecs
/// with HashMap lookup caches, making it cheap to pass to the assembler.
#[derive(Debug, Default)]
pub struct MeshModel {
    /// Ordered list of nodes (insertion order).
    pub nodes: Vec<Node>,
    /// Ordered list of elements (insertion order).
    pub elements: Vec<Element>,
    /// Named node sets.
    pub node_sets: HashMap<String, NodeSet>,
    /// Named element sets.
    pub element_sets: HashMap<String, ElementSet>,

    // Lookup caches (built lazily / invalidated on mutation)
    node_id_to_index: HashMap<u64, usize>,
    element_id_to_index: HashMap<u64, usize>,

    /// Flat (n_nodes × 3) coordinate array — built lazily.
    node_coords_cache: Option<Vec<f64>>,
}

impl MeshModel {
    pub fn new() -> Self {
        Self::default()
    }

    // -------------------------------------------------------------------------
    // Mutation
    // -------------------------------------------------------------------------

    pub fn add_node(&mut self, node: Node) -> Result<(), MeshError> {
        if self.node_id_to_index.contains_key(&node.id) {
            return Err(MeshError::DuplicateNodeId(node.id));
        }
        let idx = self.nodes.len();
        self.node_id_to_index.insert(node.id, idx);
        self.nodes.push(node);
        self.node_coords_cache = None;
        Ok(())
    }

    pub fn add_element(&mut self, elem: Element) -> Result<(), MeshError> {
        if self.element_id_to_index.contains_key(&elem.id) {
            return Err(MeshError::DuplicateElementId(elem.id));
        }
        let idx = self.elements.len();
        self.element_id_to_index.insert(elem.id, idx);
        self.elements.push(elem);
        Ok(())
    }

    pub fn add_node_set(&mut self, set: NodeSet) {
        self.node_sets.insert(set.name.clone(), set);
    }

    pub fn add_element_set(&mut self, set: ElementSet) {
        self.element_sets.insert(set.name.clone(), set);
    }

    // -------------------------------------------------------------------------
    // Lookup
    // -------------------------------------------------------------------------

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    pub fn node_index(&self, id: u64) -> Option<usize> {
        self.node_id_to_index.get(&id).copied()
    }

    pub fn element_index(&self, id: u64) -> Option<usize> {
        self.element_id_to_index.get(&id).copied()
    }

    pub fn node_by_id(&self, id: u64) -> Option<&Node> {
        self.node_id_to_index.get(&id).map(|&i| &self.nodes[i])
    }

    pub fn element_by_id(&self, id: u64) -> Option<&Element> {
        self.element_id_to_index.get(&id).map(|&i| &self.elements[i])
    }

    /// All element IDs in the named set.
    pub fn element_ids_in_set(&self, name: &str) -> Result<&[u64], MeshError> {
        self.element_sets
            .get(name)
            .map(|s| s.element_ids.as_slice())
            .ok_or_else(|| MeshError::ElementSetNotFound(name.to_string()))
    }

    // -------------------------------------------------------------------------
    // Flat arrays for Rust/Python interop
    // -------------------------------------------------------------------------

    /// Build and cache (n_nodes × 3) coordinate array in node-insertion order.
    ///
    /// Returns a `&[f64]` of length `n_nodes * 3` laid out as `[x0,y0,z0, x1,y1,z1, ...]`.
    pub fn node_coords_flat(&mut self) -> &[f64] {
        if self.node_coords_cache.is_none() {
            let mut v = Vec::with_capacity(self.nodes.len() * 3);
            for n in &self.nodes {
                v.push(n.x);
                v.push(n.y);
                v.push(n.z);
            }
            self.node_coords_cache = Some(v);
        }
        self.node_coords_cache.as_deref().unwrap()
    }

    /// Build connectivity list: for each element, the **0-based indices** into
    /// `self.nodes` (not node IDs). Returns `(connectivity, elem_type_codes)`.
    ///
    /// `connectivity[i]` is a `Vec<usize>` of node indices for element `i`.
    /// `elem_type_codes[i]` is the assembler numeric code (i32) for element `i`.
    pub fn build_connectivity_arrays(
        &self,
    ) -> (Vec<Vec<usize>>, Vec<i32>) {
        let mut connectivity = Vec::with_capacity(self.elements.len());
        let mut codes = Vec::with_capacity(self.elements.len());

        for elem in &self.elements {
            let conn: Vec<usize> = elem
                .node_ids
                .iter()
                .map(|nid| self.node_id_to_index[nid])
                .collect();
            connectivity.push(conn);
            codes.push(elem.element_type.assembler_code());
        }

        (connectivity, codes)
    }

    /// Map from node ID → 0-based index in `self.nodes`.
    pub fn node_id_to_index_map(&self) -> &HashMap<u64, usize> {
        &self.node_id_to_index
    }
}
