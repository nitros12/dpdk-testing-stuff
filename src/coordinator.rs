use capsule::{PortQueue, Mbuf, dpdk};
use rustacuda::{memory::UnifiedBuffer, prelude::{StreamFlags, Stream}, module::Module, launch};
use std::{ffi::CString, error::Error, sync::{mpsc::{Sender, Receiver, channel}, Arc}};

struct Worker {
    max_capacity: u16,
    stream: Arc<Stream>,
    result_buf: UnifiedBuffer<u32>,
    pkt_ptr_buf: UnifiedBuffer<usize>,
    mbufs: Vec<Mbuf>,
    temp_buf: Vec<*mut ()>,
}

// SAFETY: I promise to only ever use this from the thread it came from
unsafe impl Send for Worker {}

impl Worker {
    fn recv(&mut self, pq: &PortQueue) {
        pq.receive_into(&mut self.temp_buf, &mut self.mbufs, self.max_capacity);

        for (idx, pkt) in self.mbufs.iter().enumerate() {
            self.pkt_ptr_buf[idx] = get_mbuf_data(pkt).as_ptr() as usize;
        }

        // println!("pkt_ptr_buf: {:?} {:?}", self.pkt_ptr_buf.as_unified_ptr(), &self.pkt_ptr_buf[0..self.mbufs.len()]);
    }

    fn split_data(&mut self) -> (&Stream, &mut UnifiedBuffer<u32>, &mut UnifiedBuffer<usize>, usize) {
        (&self.stream, &mut self.result_buf, &mut self.pkt_ptr_buf, self.mbufs.len())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

fn make_worker(max_capacity: usize) -> Result<Worker, Box<dyn Error>> {
    let stream = Arc::new(Stream::new(StreamFlags::NON_BLOCKING, None)?);
    let result_buf = UnifiedBuffer::new(&0u32, max_capacity)?;
    let pkt_ptr_buf = UnifiedBuffer::new(&0usize, max_capacity)?;
    let mbufs = Vec::with_capacity(max_capacity);
    let temp_buf = Vec::with_capacity(max_capacity);

    Ok(Worker {
        max_capacity: max_capacity as u16,
        stream,
        result_buf,
        pkt_ptr_buf,
        mbufs,
        temp_buf,
    })
}

pub struct Coordinator {
    workers: Receiver<Worker>,
    returner: Sender<Worker>,
    module: Module,
}

fn get_mbuf_data(buf: &dpdk::Mbuf) -> &mut [u8] {
    unsafe {
        let ptr = buf.read_data_slice::<u8>(0, buf.data_len()).unwrap();
        &mut *ptr.as_ptr()
    }
}

impl Coordinator {
    pub fn new(max_capacity: usize, max_concurrency: usize) -> Result<Self, Box<dyn Error>> {
        let (returner, workers) = channel();

        for _ in 0..max_concurrency {
            returner.send(make_worker(max_capacity)?)?;
        }

        let ptx = CString::new(include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx")))?;
        let module = Module::load_from_string(&ptx)?;

        Ok(Coordinator { workers, returner, module })
    }

    pub fn process_packets(&self, pq: &PortQueue) -> Result<usize, Box<dyn Error>> {
        let mut worker = self.workers.recv()?;

        worker.recv(pq);

        let (stream, pkt_ptr_buf, result_buf, pkt_len) = worker.split_data();
        let module = &self.module;

        unsafe {
            launch!(module.perform<<<pkt_len as u32, 1, 0, stream>>>(
                pkt_ptr_buf.as_unified_ptr(),
                result_buf.as_unified_ptr(),
                pkt_len
            ))?;
        }


        let s_c = worker.stream.clone();
        let ret = self.returner.clone();
        s_c.add_callback(Box::new(move |s: Result<(), rustacuda::error::CudaError>| {
            // TODO: send packets

            // return the worker to the pool
            let _ = ret.send(worker);
        }))?;


        Ok(pkt_len)
    }

    /// make sure this is called, otherwise things will explode
    pub fn close(self) {
        drop(self.returner);
        while let Ok(v) = self.workers.recv() {
            drop(v);
        }
    }
}
