@group(0) @binding(0) var<storage, read> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read_write> archive: array<Contact>;
@group(0) @binding(2) var<storage, read_write> contact_count: array<atomic<u32>>;

fn extent() -> u32 {
    return min(atomicLoad(&contact_count[0]), arrayLength(&contacts));
}

fn work(index: u32) {
    archive[index] = contacts[index];
}

