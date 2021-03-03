use capsule::{Mbuf, PortQueue};

pub struct PortInfo<'a> {
    pub port: &'a PortQueue,
    pub transmit_queue: Vec<Mbuf>,
    pub transmitted: usize,
}

impl<'a> PortInfo<'a> {
    pub fn new(port: &'a PortQueue) -> Self {
        Self {
            port,
            transmit_queue: Vec::new(),
            transmitted: 0,
        }
    }

    pub fn add_to_queue(&mut self, mbuf: Mbuf) {
        self.transmit_queue.push(mbuf);
    }

    pub fn flush(&mut self) {
        if !self.transmit_queue.is_empty() {
            self.transmitted += self.transmit_queue.len();

            self.port.transmit_from(&mut self.transmit_queue);
        }
    }
}
