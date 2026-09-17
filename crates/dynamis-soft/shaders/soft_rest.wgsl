@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> bodies: array<SoftBody>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;

fn work(index: u32) {
    let moving = atomicExchange(&bodies[index].moving, 0u) != 0u;
    let woken = atomicExchange(&bodies[index].wake, 0u) != 0u;
    let sleeping = bodies[index].sleeping != 0u;
    if (woken && sleeping) {
        bodies[index].sleep_timer = 0.0;
        bodies[index].sleeping = 0u;
        counter_add(COUNTER_SOFT_WOKE, 1u);
    } else if (!sleeping) {
        if (moving) {
            bodies[index].sleep_timer = 0.0;
        } else {
            let timer = bodies[index].sleep_timer + params.dt;
            if (timer >= params.sleep_time) {
                bodies[index].sleep_timer = 0.0;
                bodies[index].sleeping = 1u;
                counter_add(COUNTER_SOFT_SLEPT, 1u);
            } else {
                bodies[index].sleep_timer = timer;
            }
        }
    }
    if (bodies[index].sleeping == 0u) {
        counter_add(COUNTER_SOFT_ACTIVE, 1u);
    }
}
