use std::fmt;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Entity {
    index: u32,
    generation: u32,
}

impl Entity {
    pub(crate) fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    pub fn index(self) -> u32 {
        self.index
    }

    pub fn generation(self) -> u32 {
        self.generation
    }
}

impl fmt::Debug for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entity({}:v{})", self.index, self.generation)
    }
}

#[derive(Default)]
pub(crate) struct EntityAllocator {
    free: Vec<u32>,
    generations: Vec<u32>,
}

impl EntityAllocator {
    pub(crate) fn alloc(&mut self) -> Entity {
        if let Some(index) = self.free.pop() {
            Entity::new(index, self.generations[index as usize])
        } else {
            let index = u32::try_from(self.generations.len()).expect("entity index overflow");
            self.generations.push(0);
            Entity::new(index, 0)
        }
    }

    pub(crate) fn free(&mut self, entity: Entity) {
        let slot = self
            .generations
            .get_mut(entity.index as usize)
            .expect("despawning entity with out-of-range index");
        if *slot != entity.generation {
            panic!("despawning stale entity {entity:?}");
        }
        *slot = entity.generation.wrapping_add(1);
        self.free.push(entity.index);
    }
}
