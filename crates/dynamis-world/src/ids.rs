pub(crate) struct IdSpace {
    free: Vec<u32>,
    generations: Vec<u32>,
}

impl IdSpace {
    pub(crate) const fn new() -> Self {
        Self {
            free: Vec::new(),
            generations: Vec::new(),
        }
    }

    pub(crate) fn acquire(&mut self) -> (u32, u32) {
        let id = match self.free.pop() {
            Some(id) => id,
            None => {
                self.generations.push(1);
                (self.generations.len() - 1) as u32
            }
        };
        self.generations[id as usize] += 1;
        (id, self.generations[id as usize])
    }

    pub(crate) fn release(&mut self, id: u32) {
        self.generations[id as usize] += 1;
        self.free.push(id);
    }

    pub(crate) fn generation(&self, id: u32) -> u32 {
        self.generations[id as usize]
    }

    pub(crate) fn len(&self) -> usize {
        self.generations.len()
    }
}
