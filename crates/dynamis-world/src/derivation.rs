use dynamis_gpu::StreamIdentity;

/// The facts every derived device index is a pure function of. Each field is a monotone revision of
/// one scene fact, bumped by the single authority that owns it: a store's record path, the command
/// stream the world compiles, or the world's own bookkeeping. A derivation records the facts it was
/// derived from and compares them against these, so no mutation path has to remember to invalidate
/// a derived index, and a derivation that lives in a buffer can tell when the buffer is replaced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SceneFacts {
    /// Every collider record: shape, scale, transform, material, and partition membership.
    pub(crate) colliders: u64,
    /// Every shape source's geometry: the bounds a source shape's entries are keyed from.
    pub(crate) shapes: u64,
    /// Every edit the command stream declares for an immovable body.
    pub(crate) immovable_edits: u64,
    /// Every pose the command stream declares for a body.
    pub(crate) poses: u64,
    /// Every joint's endpoints, kind, and payload.
    pub(crate) joints: u64,
    /// Every body a joint order hangs off: existence, partition, and staticness.
    pub(crate) anchors: u64,
    /// Every move of a body row.
    pub(crate) layout: u64,
    /// Every sleep or wake transition the device reported.
    pub(crate) activity: u64,
}

impl SceneFacts {
    /// Replaces the scene. Every fact moves past the value any derivation could have recorded, so
    /// every derived index acknowledges the replacement as an input it has never seen.
    pub(crate) fn replace(&mut self) {
        self.colliders += 1;
        self.shapes += 1;
        self.immovable_edits += 1;
        self.poses += 1;
        self.joints += 1;
        self.anchors += 1;
        self.layout += 1;
        self.activity += 1;
    }
}

/// The facts the immovable half of the spatial grid is derived from: the records and source
/// geometry its AABBs are built from, the edits the command stream declares for them, the counts
/// that move the grid resolution, the range it is emitted into, and the storage it lives in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ImmovableGrid {
    pub(crate) colliders: u64,
    pub(crate) shapes: u64,
    pub(crate) immovable_edits: u64,
    pub(crate) resolution: Resolution,
    pub(crate) reservation: u32,
    pub(crate) storage: GridStorage,
}

/// The facts the resting half of the spatial grid is derived from: the records and source geometry
/// every sleeping collider's AABB is built from, the poses the command stream declares, the row
/// layout its owners are read through, the sleep state that selects it, the resolution its entries
/// are keyed at, and the storage it lives in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RestingGrid {
    pub(crate) colliders: u64,
    pub(crate) shapes: u64,
    pub(crate) poses: u64,
    pub(crate) layout: u64,
    pub(crate) activity: u64,
    pub(crate) resolution: (u32, u32),
    pub(crate) storage: GridStorage,
}

/// The facts a joint order is derived from: the joints themselves and the bodies they hang off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct JointOrder {
    pub(crate) joints: u64,
    pub(crate) anchors: u64,
}

impl JointOrder {
    pub(crate) fn of(facts: &SceneFacts) -> Self {
        Self {
            joints: facts.joints,
            anchors: facts.anchors,
        }
    }
}

/// The scene facts a grid resolution is derived from. Every one of them can move the grid
/// resolution, and an index derived at another resolution must be derived again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Resolution {
    pub(crate) colliders: u32,
    pub(crate) movable_colliders: u32,
    pub(crate) particles: u32,
}

/// The storage the spatial grid entries live in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GridStorage {
    pub(crate) keys: StreamIdentity,
    pub(crate) order: StreamIdentity,
    pub(crate) entries: StreamIdentity,
}

/// The storage a joint order is uploaded into: the row run, the layers, the components, and the
/// workgroup batches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct JointOrderStorage {
    pub(crate) rows: StreamIdentity,
    pub(crate) layers: StreamIdentity,
    pub(crate) components: StreamIdentity,
    pub(crate) batches: StreamIdentity,
}
