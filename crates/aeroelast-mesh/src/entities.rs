/// Element type classification — mirrors Python `ElementType` / VTK cell codes.
///
/// Values are the same numeric codes used by the Python assembler
/// (`code` field in `materials_list`) so `PyMeshAssembler` can consume them directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ElementType {
    // Shell / surface
    Triangle3  = 3,   // MITC3
    Triangle6  = 6,   // quadratic triangle (future)
    Quad4      = 4,   // MITC4
    Quad8      = 8,   // serendipity quad
    Quad9      = 9,   // Lagrange quad

    // Shell composite variants (codes used by PyMeshAssembler)
    CompTri3   = 33,  // MITC3 composite
    CompQuad4  = 44,  // MITC4 composite

    // Solid 3D
    Tetra4     = 14,
    Tetra10    = 24,
    Hexa8      = 18,
    Hexa20     = 28,
    Wedge6     = 16,
    Wedge15    = 26,
    Pyramid5   = 15,
    Pyramid13  = 25,
}

impl ElementType {
    pub fn node_count(self) -> usize {
        match self {
            Self::Triangle3 | Self::CompTri3 => 3,
            Self::Triangle6 => 6,
            Self::Quad4 | Self::CompQuad4 => 4,
            Self::Quad8 => 8,
            Self::Quad9 => 9,
            Self::Tetra4 => 4,
            Self::Tetra10 => 10,
            Self::Hexa8 => 8,
            Self::Hexa20 => 20,
            Self::Wedge6 => 6,
            Self::Wedge15 => 15,
            Self::Pyramid5 => 5,
            Self::Pyramid13 => 13,
        }
    }

    pub fn assembler_code(self) -> i32 {
        self as i32
    }

    pub fn is_composite(self) -> bool {
        matches!(self, Self::CompTri3 | Self::CompQuad4)
    }

    pub fn is_shell(self) -> bool {
        matches!(
            self,
            Self::Triangle3 | Self::Triangle6 | Self::Quad4 | Self::Quad8 | Self::Quad9
            | Self::CompTri3 | Self::CompQuad4
        )
    }
}

/// A single node in 3D space with a unique ID.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: u64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Node {
    pub fn new(id: u64, x: f64, y: f64, z: f64) -> Self {
        Self { id, x, y, z }
    }

    pub fn coords(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

/// A mesh element defined by node IDs and type.
#[derive(Debug, Clone)]
pub struct Element {
    pub id: u64,
    pub node_ids: Vec<u64>,
    pub element_type: ElementType,
}

impl Element {
    pub fn new(id: u64, node_ids: Vec<u64>, element_type: ElementType) -> Self {
        Self { id, node_ids, element_type }
    }

    pub fn node_count(&self) -> usize {
        self.node_ids.len()
    }
}

/// A named set of nodes.
#[derive(Debug, Clone, Default)]
pub struct NodeSet {
    pub name: String,
    pub node_ids: Vec<u64>,
}

impl NodeSet {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), node_ids: Vec::new() }
    }

    pub fn with_ids(name: impl Into<String>, node_ids: Vec<u64>) -> Self {
        Self { name: name.into(), node_ids }
    }

    pub fn add(&mut self, id: u64) {
        self.node_ids.push(id);
    }
}

/// A named set of elements.
#[derive(Debug, Clone, Default)]
pub struct ElementSet {
    pub name: String,
    pub element_ids: Vec<u64>,
}

impl ElementSet {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), element_ids: Vec::new() }
    }

    pub fn with_ids(name: impl Into<String>, element_ids: Vec<u64>) -> Self {
        Self { name: name.into(), element_ids }
    }

    pub fn add(&mut self, id: u64) {
        self.element_ids.push(id);
    }
}
