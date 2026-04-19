pub mod mitc3;
pub mod mitc4;
pub mod quad;
pub mod solid;

/// Result of a batch element computation.
pub struct BatchResult {
    /// Flat array of element matrices: [n_elem × n_dof × n_dof]
    pub matrices: Vec<f64>,
    /// Number of elements
    pub n_elem: usize,
    /// DOFs per element
    pub n_dof: usize,
}

/// Result of a batch force computation.
pub struct BatchForceResult {
    /// Flat array of element force vectors: [n_elem × n_dof]
    pub forces: Vec<f64>,
    pub n_elem: usize,
    pub n_dof: usize,
}
