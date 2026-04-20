/// Element type classification — maps to PyMeshAssembler numeric codes.
///
/// The `assembler_code()` values match the codes accepted by `PyMeshAssembler.__init__`
/// and `PyMeshAssembler.from_mesh_model()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    // Shell / surface
    Triangle3,    // MITC3          — assembler code 3
    Triangle6,    // quadratic tri  — assembler code 108 (treated as Quad8 placeholder, future)
    Quad4,        // MITC4          — assembler code 4
    Quad8,        // serendipity    — assembler code 108
    Quad9,        // Lagrange quad  — assembler code 109

    // Shell composite variants
    CompTri3,     // MITC3 composite  — assembler code 33
    CompQuad4,    // MITC4 composite  — assembler code 44

    // Solid 3D
    Tetra4,       // assembler code 304
    Tetra10,      // assembler code 310
    Hexa8,        // assembler code 208
    Hexa20,       // assembler code 220
    Wedge6,       // assembler code 306
    Wedge15,      // assembler code 315
    Pyramid5,     // assembler code 305
    Pyramid13,    // assembler code 313
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

    /// Returns the numeric code expected by `PyMeshAssembler`.
    pub fn assembler_code(self) -> i32 {
        match self {
            Self::Triangle3  => 3,
            Self::CompTri3   => 33,
            Self::Quad4      => 4,
            Self::CompQuad4  => 44,
            Self::Triangle6  => 108, // placeholder — not yet in assembler
            Self::Quad8      => 108,
            Self::Quad9      => 109,
            Self::Hexa8      => 208,
            Self::Hexa20     => 220,
            Self::Tetra4     => 304,
            Self::Tetra10    => 310,
            Self::Wedge6     => 306,
            Self::Wedge15    => 315,
            Self::Pyramid5   => 305,
            Self::Pyramid13  => 313,
        }
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
