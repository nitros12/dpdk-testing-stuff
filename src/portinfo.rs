use std::sync::{
    atomic::{AtomicUsize, Ordering::Relaxed},
    Arc,
};

use capsule::{Mbuf, PortQueue};

pub struct PortInfo<'a> {
    pub port: &'a PortQueue,
    pub transmit_queue: Vec<Mbuf>,
    pub counter: Arc<AtomicUsize>,
}

impl<'a> PortInfo<'a> {
    pub fn new(port: &'a PortQueue, counter: Arc<AtomicUsize>) -> Self {
        Self {
            port,
            transmit_queue: Vec::new(),
            counter,
        }
    }

    pub fn add_to_queue(&mut self, mbuf: Mbuf) {
        self.transmit_queue.push(mbuf);
    }

    pub fn flush(&mut self) {
        if !self.transmit_queue.is_empty() {
            self.counter.fetch_add(self.transmit_queue.len(), Relaxed);

            self.port.transmit_from(&mut self.transmit_queue);
        }
    }
}
