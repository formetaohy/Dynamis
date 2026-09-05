use crate::component::{ComponentMeta, meta_by_index};
use crate::entity::Entity;
use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ptr::NonNull;

pub(crate) struct Column {
    ptr: NonNull<u8>,
    layout: Layout,
    element_size: usize,
    capacity: usize,
    len: usize,
    drop: unsafe fn(*mut u8),
}

impl Column {
    fn new(meta: &ComponentMeta) -> Self {
        let layout = Layout::from_size_align(0, meta.layout.align()).expect("zero column layout");
        Self {
            ptr: NonNull::dangling(),
            layout,
            element_size: meta.layout.size(),
            capacity: 0,
            len: 0,
            drop: meta.drop,
        }
    }

    fn reserve(&mut self, extra: usize) {
        let needed = self.len + extra;
        if needed <= self.capacity {
            return;
        }
        let target = needed.max(self.capacity.saturating_mul(2));
        let new_layout = Layout::from_size_align(target * self.element_size, self.layout.align())
            .expect("column layout overflow");
        let new_ptr = unsafe { alloc(new_layout) };
        let new_ptr = NonNull::new(new_ptr).unwrap_or_else(|| handle_alloc_error(new_layout));
        if self.capacity > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.ptr.as_ptr(),
                    new_ptr.as_ptr(),
                    self.len * self.element_size,
                );
                dealloc(self.ptr.as_ptr(), self.layout);
            }
        }
        self.ptr = new_ptr;
        self.layout = new_layout;
        self.capacity = target;
    }

    pub(crate) fn get(&self, row: usize) -> &[u8] {
        assert!(row < self.len, "row out of bounds");
        let start = row * self.element_size;
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().add(start), self.element_size) }
    }

    pub(crate) fn get_mut(&mut self, row: usize) -> &mut [u8] {
        assert!(row < self.len, "row out of bounds");
        let start = row * self.element_size;
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr().add(start), self.element_size) }
    }

    pub(crate) fn push(&mut self, bytes: &[u8]) {
        assert_eq!(
            bytes.len(),
            self.element_size,
            "column element size mismatch"
        );
        self.reserve(1);
        let start = self.len * self.element_size;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.ptr.as_ptr().add(start),
                self.element_size,
            );
        }
        self.len += 1;
    }

    pub(crate) fn swap_remove(&mut self, row: usize) -> Vec<u8> {
        assert!(row < self.len, "row out of bounds");
        let mut bytes = vec![0u8; self.element_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr.as_ptr().add(row * self.element_size),
                bytes.as_mut_ptr(),
                self.element_size,
            );
            if row + 1 < self.len {
                std::ptr::copy_nonoverlapping(
                    self.ptr.as_ptr().add((self.len - 1) * self.element_size),
                    self.ptr.as_ptr().add(row * self.element_size),
                    self.element_size,
                );
            }
            self.len -= 1;
        }
        bytes
    }

    pub(crate) fn drop_rows(&mut self) {
        unsafe {
            for row in 0..self.len {
                let ptr = self.ptr.as_ptr().add(row * self.element_size);
                (self.drop)(ptr);
            }
        }
        self.len = 0;
    }
}

impl Drop for Column {
    fn drop(&mut self) {
        self.drop_rows();
        if self.capacity > 0 {
            unsafe {
                dealloc(self.ptr.as_ptr(), self.layout);
            }
        }
    }
}

pub struct Archetype {
    key: Box<[u32]>,
    rows: Vec<Entity>,
    columns: HashMap<u32, UnsafeCell<Column>>,
}

impl Archetype {
    pub(crate) fn new(key: Box<[u32]>) -> Self {
        Self {
            key,
            rows: Vec::new(),
            columns: HashMap::new(),
        }
    }

    pub(crate) fn key(&self) -> &[u32] {
        &self.key
    }

    pub(crate) fn count(&self) -> usize {
        self.rows.len()
    }

    pub(crate) fn row_entity(&self, row: usize) -> Entity {
        *self.rows.get(row).expect("row out of bounds")
    }

    pub(crate) fn push_row(&mut self, entity: Entity, parts: &[(u32, Vec<u8>)]) {
        self.rows.push(entity);
        for (index, bytes) in parts {
            let column = self
                .columns
                .entry(*index)
                .or_insert_with(|| UnsafeCell::new(Column::new(&meta_by_index(*index))));
            column.get_mut().push(bytes);
        }
    }

    pub(crate) fn remove_row(&mut self, row: usize) -> (Entity, Vec<(u32, Vec<u8>)>) {
        let entity = self.row_entity(row);
        let mut parts = Vec::with_capacity(self.columns.len());
        for (index, column) in &mut self.columns {
            parts.push((*index, column.get_mut().swap_remove(row)));
        }
        self.rows.swap_remove(row);
        (entity, parts)
    }

    pub(crate) fn delete_row(&mut self, row: usize) -> Entity {
        let entity = self.row_entity(row);
        for (index, column) in &mut self.columns {
            let bytes = column.get_mut().swap_remove(row);
            let meta = meta_by_index(*index);
            unsafe {
                (meta.drop)(bytes.as_ptr().cast_mut());
            }
        }
        self.rows.swap_remove(row);
        entity
    }

    pub(crate) fn column(&self, index: u32) -> Option<&Column> {
        self.columns
            .get(&index)
            .map(|column| unsafe { &*column.get() })
    }

    pub(crate) fn column_raw(&self, index: u32) -> Option<*mut Column> {
        self.columns.get(&index).map(|column| column.get())
    }

    pub(crate) fn column_mut(&mut self, index: u32) -> Option<&mut Column> {
        self.columns
            .get(&index)
            .map(|column| unsafe { &mut *column.get() })
    }

    pub(crate) fn contains(&self, index: u32) -> bool {
        self.columns.contains_key(&index)
    }

    pub(crate) fn drop_all(&mut self) {
        for column in self.columns.values_mut() {
            column.get_mut().drop_rows();
        }
        self.rows.clear();
    }
}

impl Drop for Archetype {
    fn drop(&mut self) {
        self.drop_all();
    }
}
