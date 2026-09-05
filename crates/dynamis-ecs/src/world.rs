use crate::archetype::Archetype;
use crate::bundle::ComponentBundle;
use crate::component::Component;
use crate::component::{component_meta, meta_by_index};
use crate::entity::{Entity, EntityAllocator};
use crate::query::{Query, QueryItem};
use std::collections::{BTreeMap, HashMap};

#[derive(Copy, Clone)]
struct Location {
    archetype: usize,
    row: usize,
}

pub struct World {
    entities: EntityAllocator,
    locations: HashMap<Entity, Location>,
    archetypes: Vec<Archetype>,
    keys: BTreeMap<Box<[u32]>, usize>,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    pub fn new() -> Self {
        Self {
            entities: EntityAllocator::default(),
            locations: HashMap::new(),
            archetypes: Vec::new(),
            keys: BTreeMap::new(),
        }
    }

    fn archetype_index(&mut self, key: &[u32]) -> usize {
        if let Some(&index) = self.keys.get(key) {
            return index;
        }
        let index = self.archetypes.len();
        self.archetypes
            .push(Archetype::new(key.to_vec().into_boxed_slice()));
        self.keys.insert(key.to_vec().into_boxed_slice(), index);
        index
    }

    fn location(&self, entity: Entity) -> Location {
        *self
            .locations
            .get(&entity)
            .unwrap_or_else(|| panic!("entity {entity:?} is not alive"))
    }

    fn sort_indices(mut indices: Vec<u32>) -> Vec<u32> {
        indices.sort_unstable();
        if indices.windows(2).any(|pair| pair[0] == pair[1]) {
            panic!("spawning or inserting duplicate component types");
        }
        indices
    }

    fn relocate_swapped(&mut self, archetype: usize, row: usize) {
        let swapped = self
            .archetypes
            .get(archetype)
            .and_then(|archetype| (row < archetype.count()).then(|| archetype.row_entity(row)));
        if let Some(swapped) = swapped {
            self.locations
                .get_mut(&swapped)
                .expect("swapped entity missing")
                .row = row;
        }
    }

    pub fn spawn(&mut self, bundle: impl ComponentBundle) -> Entity {
        let entity = self.entities.alloc();
        let parts = bundle.into_parts();
        let indices: Vec<u32> = parts.iter().map(|(index, _)| *index).collect();
        let key = Self::sort_indices(indices).into_boxed_slice();
        let archetype = self.archetype_index(&key);
        let row = self.archetypes[archetype].count();
        self.archetypes[archetype].push_row(entity, &parts);
        self.locations.insert(entity, Location { archetype, row });
        entity
    }

    pub fn despawn(&mut self, entity: Entity) {
        let location = self.location(entity);
        let removed = self.archetypes[location.archetype].delete_row(location.row);
        debug_assert_eq!(removed, entity);
        self.relocate_swapped(location.archetype, location.row);
        self.locations.remove(&entity);
        self.entities.free(entity);
    }

    pub fn insert(&mut self, entity: Entity, bundle: impl ComponentBundle) {
        let location = self.location(entity);
        let new_parts = bundle.into_parts();
        let additions: Vec<u32> = new_parts.iter().map(|(index, _)| *index).collect();
        let additions = Self::sort_indices(additions);
        let old_key = self.archetypes[location.archetype].key().to_vec();
        for index in &additions {
            if old_key.binary_search(index).is_ok() {
                panic!("entity {entity:?} already has a component resolved to index {index}");
            }
        }
        let mut new_key = old_key;
        new_key.extend(additions.iter().copied());
        new_key.sort_unstable();
        let target = self.archetype_index(&new_key);

        let (moved, moved_parts) = self.archetypes[location.archetype].remove_row(location.row);
        self.relocate_swapped(location.archetype, location.row);

        let mut all_parts = moved_parts;
        all_parts.extend(new_parts);
        let row = self.archetypes[target].count();
        self.archetypes[target].push_row(moved, &all_parts);
        let slot = self
            .locations
            .get_mut(&moved)
            .expect("moved entity missing");
        slot.archetype = target;
        slot.row = row;
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) {
        let (index, _) = component_meta::<T>();
        let location = self.location(entity);
        let old_key = self.archetypes[location.archetype].key();
        if old_key.binary_search(&index).is_err() {
            panic!("entity {entity:?} does not carry the component being removed");
        }
        let new_key: Vec<u32> = old_key
            .iter()
            .copied()
            .filter(|existing| *existing != index)
            .collect();
        let target = self.archetype_index(&new_key);

        let (moved, moved_parts) = self.archetypes[location.archetype].remove_row(location.row);
        self.relocate_swapped(location.archetype, location.row);

        let mut dropped = None;
        let mut kept = Vec::new();
        for (part_index, bytes) in moved_parts {
            if part_index == index {
                dropped = Some(bytes);
            } else {
                kept.push((part_index, bytes));
            }
        }
        let dropped = dropped.expect("removed component data missing");
        let meta = meta_by_index(index);
        unsafe {
            (meta.drop)(dropped.as_ptr().cast_mut());
        }

        let row = self.archetypes[target].count();
        self.archetypes[target].push_row(moved, &kept);
        let slot = self
            .locations
            .get_mut(&moved)
            .expect("moved entity missing");
        slot.archetype = target;
        slot.row = row;
    }

    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        let location = *self.locations.get(&entity)?;
        let (index, _) = component_meta::<T>();
        let bytes = self.archetypes[location.archetype]
            .column(index)?
            .get(location.row);
        Some(unsafe { &*bytes.as_ptr().cast::<T>() })
    }

    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        let location = *self.locations.get(&entity)?;
        let (index, _) = component_meta::<T>();
        let bytes = self.archetypes[location.archetype]
            .column_mut(index)?
            .get_mut(location.row);
        Some(unsafe { &mut *bytes.as_mut_ptr().cast::<T>() })
    }

    pub fn has<T: Component>(&self, entity: Entity) -> bool {
        let (index, _) = component_meta::<T>();
        self.locations
            .get(&entity)
            .and_then(|location| self.archetypes[location.archetype].column(index))
            .is_some()
    }

    pub fn query<Q: QueryItem>(&self) -> Query<'_, Q> {
        Query::new(self)
    }

    pub(crate) fn archetypes(&self) -> &[Archetype] {
        &self.archetypes
    }

    pub fn alive(&self, entity: Entity) -> bool {
        self.locations.contains_key(&entity)
    }

    pub fn len(&self) -> usize {
        self.locations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.locations.is_empty()
    }
}
