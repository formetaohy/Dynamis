use crate::archetype::Archetype;
use crate::component::Component;
use crate::world::World;
use std::marker::PhantomData;

pub struct Query<'w, Q: QueryItem> {
    world: &'w World,
    marker: PhantomData<Q>,
}

impl<'w, Q: QueryItem> Query<'w, Q> {
    pub(crate) fn new(world: &'w World) -> Self {
        Self {
            world,
            marker: PhantomData,
        }
    }

    pub fn for_each(self, mut f: impl FnMut(Q::Item<'w>)) {
        let accesses = Q::accesses();
        validate_accesses(&accesses);
        for archetype in self.world.archetypes() {
            if accesses.iter().all(|(index, _)| archetype.contains(*index)) {
                for row in 0..archetype.count() {
                    f(Q::fetch(archetype, row));
                }
            }
        }
    }

    pub fn count(&self) -> usize {
        let accesses = Q::accesses();
        self.world
            .archetypes()
            .iter()
            .filter(|archetype| accesses.iter().all(|(index, _)| archetype.contains(*index)))
            .map(|archetype| archetype.count())
            .sum()
    }
}

pub trait QueryItem {
    type Item<'a>;
    fn accesses() -> Vec<(u32, bool)>;
    fn fetch<'a>(archetype: &'a Archetype, row: usize) -> Self::Item<'a>;
}

fn validate_accesses(accesses: &[(u32, bool)]) {
    for (position, (index, mutable)) in accesses.iter().enumerate() {
        for (other, other_mutable) in &accesses[position + 1..] {
            if index == other && (*mutable || *other_mutable) {
                panic!("conflicting component accesses in query");
            }
        }
    }
}

macro_rules! query_ref {
    (ro, $typ:ident, $lt:lifetime) => {
        &$lt $typ
    };
    (rw, $typ:ident, $lt:lifetime) => {
        &$lt mut $typ
    };
}

macro_rules! query_is_mut {
    (ro) => {
        false
    };
    (rw) => {
        true
    };
}

macro_rules! fetch_one {
    (ro, $typ:ident, $archetype:ident, $row:ident) => {
        unsafe {
            &*$archetype
                .column(crate::component_meta::<$typ>().0)
                .expect("query found a row without the required column")
                .get($row)
                .as_ptr()
                .cast::<$typ>()
        }
    };
    (rw, $typ:ident, $archetype:ident, $row:ident) => {
        unsafe {
            &mut *(&mut *$archetype
                .column_raw(crate::component_meta::<$typ>().0)
                .expect("query found a row without the required column"))
                .get_mut($row)
                .as_mut_ptr()
                .cast::<$typ>()
        }
    };
}

macro_rules! impl_query_item {
    ($( $mutability:ident $field:ident : $typ:ident ),+ $(,)?) => {
        impl<'q, $($typ: Component),+> QueryItem for ($( query_ref!($mutability, $typ, 'q), )+) {
            type Item<'a> = ($( query_ref!($mutability, $typ, 'a), )+);

            fn accesses() -> Vec<(u32, bool)> {
                vec![$( (crate::component_meta::<$typ>().0, query_is_mut!($mutability)) ),+]
            }

            fn fetch<'a>(archetype: &'a Archetype, row: usize) -> Self::Item<'a> {
                ($( fetch_one!($mutability, $typ, archetype, row), )+)
            }
        }
    };
}

impl<A: Component> QueryItem for &A {
    type Item<'a> = &'a A;

    fn accesses() -> Vec<(u32, bool)> {
        vec![(crate::component_meta::<A>().0, false)]
    }

    fn fetch<'a>(archetype: &'a Archetype, row: usize) -> Self::Item<'a> {
        fetch_one!(ro, A, archetype, row)
    }
}

impl<A: Component> QueryItem for &mut A {
    type Item<'a> = &'a mut A;

    fn accesses() -> Vec<(u32, bool)> {
        vec![(crate::component_meta::<A>().0, true)]
    }

    fn fetch<'a>(archetype: &'a Archetype, row: usize) -> Self::Item<'a> {
        fetch_one!(rw, A, archetype, row)
    }
}

impl_query_item!(ro a: A);
impl_query_item!(rw a: A);
impl_query_item!(ro a: A, ro b: B);
impl_query_item!(ro a: A, rw b: B);
impl_query_item!(rw a: A, ro b: B);
impl_query_item!(rw a: A, rw b: B);
impl_query_item!(ro a: A, ro b: B, ro c: C);
impl_query_item!(ro a: A, ro b: B, rw c: C);
impl_query_item!(ro a: A, rw b: B, ro c: C);
impl_query_item!(rw a: A, ro b: B, ro c: C);
impl_query_item!(rw a: A, ro b: B, rw c: C);
impl_query_item!(rw a: A, rw b: B, ro c: C);
impl_query_item!(ro a: A, rw b: B, rw c: C);
impl_query_item!(rw a: A, rw b: B, rw c: C);
impl_query_item!(ro a: A, ro b: B, ro c: C, rw d: D);
impl_query_item!(ro a: A, ro b: B, rw c: C, rw d: D);
impl_query_item!(rw a: A, rw b: B, rw c: C, ro d: D);
impl_query_item!(rw a: A, rw b: B, ro c: C, ro d: D);
impl_query_item!(rw a: A, ro b: B, ro c: C, ro d: D);
impl_query_item!(ro a: A, rw b: B, rw c: C, rw d: D);
impl_query_item!(rw a: A, rw b: B, rw c: C, rw d: D);
