fn joined_by_joint(first_body: u32, second_body: u32) -> bool {
    let count = min(atomicLoad(&joint_count[0]), arrayLength(&joint_major));
    if (count == 0u) {
        return false;
    }
    let a = min(first_body, second_body);
    let b = max(first_body, second_body);
    var lo = 0u;
    var hi = count;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (joint_major[mid] < a || (joint_major[mid] == a && joint_minor[mid] < b)) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo < count && joint_major[lo] == a && joint_minor[lo] == b;
}
