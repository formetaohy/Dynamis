use crate::body::BodyHandle;

#[derive(Clone, Copy, Debug)]
pub struct QueryFilter {
    pub group: u32,
    pub mask: u32,
    pub ignore_sensors: bool,
    pub ignore_sleeping: bool,
    pub ignore_static: bool,
    pub ignore_kinematic: bool,
    pub exclude: Option<BodyHandle>,
    pub include: Option<BodyHandle>,
    pub max_hits: u32,
}

impl Default for QueryFilter {
    fn default() -> Self {
        Self {
            group: 0,
            mask: u32::MAX,
            ignore_sensors: true,
            ignore_sleeping: false,
            ignore_static: false,
            ignore_kinematic: false,
            exclude: None,
            include: None,
            max_hits: 4,
        }
    }
}
