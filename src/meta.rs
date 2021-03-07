use rustacuda::memory::DeviceCopy;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Action {
    Emit = 1,
    Drop = 0,
}

unsafe impl DeviceCopy for Action {}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Metadata {
    pub output_port: u32,
    pub output_action: Action,
    pub packet_length: u32,
    pub input_port: u32,
}

impl Metadata {
    pub fn new(packet_length: u32, input_port: u32) -> Self {
        Self {
            output_port: 0,
            output_action: Action::Drop,
            packet_length,
            input_port,
        }
    }
}

impl Default for Metadata {
    fn default() -> Self {
        Self {
            output_port: 0,
            output_action: Action::Drop,
            packet_length: 0,
            input_port: 0,
        }
    }
}

unsafe impl DeviceCopy for Metadata {}
