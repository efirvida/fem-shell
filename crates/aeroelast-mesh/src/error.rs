use thiserror::Error;

#[derive(Debug, Error)]
pub enum MeshError {
    #[error("Duplicate node ID {0}")]
    DuplicateNodeId(u64),

    #[error("Duplicate element ID {0}")]
    DuplicateElementId(u64),

    #[error("Node ID {0} not found")]
    NodeNotFound(u64),

    #[error("Element ID {0} not found")]
    ElementNotFound(u64),

    #[error("Element set '{0}' not found")]
    ElementSetNotFound(String),

    #[error("I/O error: {0}")]
    Io(String),

    #[error("HDF5 error: {0}")]
    Hdf5(String),
}
