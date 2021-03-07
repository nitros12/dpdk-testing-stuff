use capsule::{dpdk, Mbuf, PortQueue};
use rustacuda::{
    launch,
    memory::{UnifiedBuffer, UnifiedPointer},
    module::Module,
    prelude::{Stream, StreamFlags},
};
use std::{
    error::Error,
    ffi::CString,
    mem::ManuallyDrop,
    sync::{
        atomic::{AtomicUsize, Ordering::Relaxed},
        mpsc::{sync_channel, Receiver, SyncSender},
        Arc, Mutex,
    },
};

use crate::meta::{Action, Metadata};
use crate::portinfo::PortInfo;

type UnifiedBufferM<T> = ManuallyDrop<UnifiedBuffer<T>>;

struct WorkerBufs {
    pkt_ptrs: UnifiedBufferM<UnifiedPointer<u8>>,
    metas: UnifiedBufferM<Metadata>,
    lengths_in: UnifiedBufferM<usize>,
    lengths_out: UnifiedBufferM<usize>,
    out_offsets: UnifiedBufferM<usize>,
}

struct WorkerPorts<'a> {
    ports: Vec<PortInfo<'a>>,
    drop_queue: Vec<Mbuf>,
    dropped_counter: Arc<AtomicUsize>,
}

struct Worker<'a> {
    max_capacity: u16,
    stream: Arc<Stream>,

    bufs: WorkerBufs,

    ports: WorkerPorts<'a>,

    temp_buf: Vec<*mut ()>,
    mbufs: Vec<Mbuf>,
    num_packets: usize,
}

// SAFETY: we only every access this from one thread at a time
unsafe impl<'a> Send for Worker<'a> {}

fn get_mbuf_data(buf: &mut dpdk::Mbuf) -> &mut [u8] {
    unsafe {
        let ptr = buf.read_data_slice::<u8>(0, buf.data_len()).unwrap();
        &mut *ptr.as_ptr()
    }
}

impl<'a> Worker<'a> {
    fn recv(&mut self, pq: &PortQueue, port_idx: u32) {
        pq.receive_into(&mut self.temp_buf, &mut self.mbufs, self.max_capacity);
        self.num_packets = self.mbufs.len();

        for (idx, pkt) in self.mbufs.iter_mut().enumerate() {
            let mbuf_data = get_mbuf_data(pkt);
            let length = mbuf_data.len();
            self.bufs.pkt_ptrs[idx] = unsafe { UnifiedPointer::wrap(mbuf_data.as_mut_ptr()) };
            self.bufs.metas[idx] = Metadata::new(length as u32, port_idx);
            self.bufs.lengths_in[idx] = length;
        }

        // println!("pkt_ptr_buf: {:?} {:?}", self.pkt_ptr_buf.as_unified_ptr(), &self.pkt_ptr_buf[0..self.mbufs.len()]);
    }

    fn split(&mut self) -> (&mut Vec<Mbuf>, &mut WorkerBufs, &mut WorkerPorts<'a>) {
        (&mut self.mbufs, &mut self.bufs, &mut self.ports)
    }

    fn flush(&mut self) {
        for port in &mut self.ports.ports {
            port.flush();
        }

        if !self.ports.drop_queue.is_empty() {
            self.ports
                .dropped_counter
                .fetch_add(self.ports.drop_queue.len(), Relaxed);

            Mbuf::free_bulk(std::mem::replace(&mut self.ports.drop_queue, Vec::new()));
        }
    }
}

impl<'a> Drop for Worker<'a> {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

fn make_worker<'a>(
    ports: &[&'a PortQueue],
    counters: &[Arc<AtomicUsize>],
    dropped_counter: Arc<AtomicUsize>,
    max_capacity: usize,
) -> Result<Worker<'a>, Box<dyn Error>> {
    let ports = ports
        .iter()
        .zip(counters.iter())
        .map(|(p, c)| PortInfo::new(*p, c.clone()))
        .collect();

    let stream = Arc::new(Stream::new(StreamFlags::NON_BLOCKING, None)?);

    let pkt_ptrs = ManuallyDrop::new(UnifiedBuffer::new(&UnifiedPointer::null(), max_capacity)?);
    let metas = ManuallyDrop::new(UnifiedBuffer::new(&Metadata::default(), max_capacity)?);
    let lengths_in = ManuallyDrop::new(UnifiedBuffer::new(&0usize, max_capacity)?);
    let lengths_out = ManuallyDrop::new(UnifiedBuffer::new(&0usize, max_capacity)?);
    let out_offsets = ManuallyDrop::new(UnifiedBuffer::new(&0usize, max_capacity)?);

    let temp_buf = Vec::with_capacity(max_capacity);
    let mbufs = Vec::with_capacity(max_capacity);
    let drop_queue = Vec::new();

    Ok(Worker {
        ports: WorkerPorts {
            ports,
            drop_queue,
            dropped_counter,
        },
        max_capacity: max_capacity as u16,
        stream,
        bufs: WorkerBufs {
            pkt_ptrs,
            metas,
            lengths_in,
            lengths_out,
            out_offsets,
        },
        temp_buf,
        mbufs,
        num_packets: 0,
    })
}

pub struct Coordinator<'a> {
    worker_map: Arc<Vec<Mutex<Worker<'a>>>>,
    workers: Receiver<usize>,
    returner: SyncSender<usize>,
    num_workers: usize,
    module: Module,
    pub counters: Vec<Arc<AtomicUsize>>,
    pub drop_counter: Arc<AtomicUsize>,
}

impl<'a> Coordinator<'a> {
    pub fn new(
        ports: &'a [&'a PortQueue],
        max_capacity: usize,
        max_concurrency: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let (returner, workers) = sync_channel(max_concurrency);
        let counters: Vec<_> = ports
            .iter()
            .map(|_| Arc::new(AtomicUsize::new(0)))
            .collect();
        let drop_counter = Arc::new(AtomicUsize::new(0));

        let mut worker_map = Vec::new();

        for idx in 0..max_concurrency {
            worker_map.push(Mutex::new(make_worker(
                ports,
                &counters,
                drop_counter.clone(),
                max_capacity,
            )?));
            returner.send(idx)?;
        }

        let ptx = CString::new(include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx")))?;
        let module = Module::load_from_string(&ptx)?;

        Ok(Coordinator {
            worker_map: Arc::new(worker_map),
            workers,
            returner,
            num_workers: max_concurrency,
            module,
            counters,
            drop_counter,
        })
    }

    pub fn process_packets(&self, pq: &PortQueue, port_idx: u32) -> Result<usize, Box<dyn Error>> {
        let worker_idx = self.workers.recv()?;
        let mut worker = self.worker_map[worker_idx].lock().unwrap();

        worker.recv(pq, port_idx);

        if worker.num_packets == 0 {
            let _  = self.returner.send(worker_idx);
            return Ok(0);
        }

        // let (stream, pkt_ptr_buf, result_buf, pkt_len) = worker.split_data();
        let num_packets = worker.num_packets;
        let stream = worker.stream.clone();
        let module = &self.module;

        unsafe {
            launch!(module.p4_process<<<num_packets as u32, 1, 0, stream>>>(
                worker.bufs.pkt_ptrs.as_unified_ptr(),
                worker.bufs.metas.as_unified_ptr(),
                worker.bufs.lengths_in.as_unified_ptr(),
                worker.bufs.lengths_out.as_unified_ptr(),
                worker.bufs.out_offsets.as_unified_ptr(),
                num_packets,
                port_idx as u64
            ))?;
        }

        let s_c = worker.stream.clone();
        let ret = self.returner.clone();
        drop(worker);
        let wm = self.worker_map.clone();
        s_c.add_callback(Box::new(
            move |s: Result<(), rustacuda::error::CudaError>| {
                if let Err(e) = s {
                    eprintln!("uh oh {:?}", e);
                    let _ = ret.send(worker_idx);
                }

                let mut worker = wm[worker_idx].lock().unwrap();
                let (mbufs, bufs, ports) = worker.split();

                for (i, mut pkt) in mbufs.drain(..).enumerate() {
                    let length_in = bufs.lengths_in[i];
                    let length_out = bufs.lengths_out[i];
                    let offset_out = bufs.out_offsets[i];
                    let meta = bufs.metas[i];

                    match length_out.cmp(&length_out) {
                        std::cmp::Ordering::Less => {
                            // packet is smaller:
                            // in:  [HHHHDDDDD]
                            // out: [0HHHDDDDD]

                            let _ = pkt.shrink(0, offset_out as usize);
                        }
                        std::cmp::Ordering::Greater => {
                            // packet is migger
                            // in:  [HHHHDDDDD]
                            // out: [HHHHDDDDD]D
                            // (I hope this is fine)
                            let _ =
                                pkt.extend(length_in as usize, (length_out - length_in) as usize);
                        }
                        std::cmp::Ordering::Equal => {}
                    }

                    match meta.output_action {
                        Action::Emit => {
                            ports.ports[meta.output_port as usize].add_to_queue(pkt);
                        }
                        Action::Drop => {
                            ports.drop_queue.push(pkt);
                        }
                    }
                }

                worker.flush();

                // return the worker to the pool
                let _ = ret.send(worker_idx);
            },
        ))?;

        Ok(num_packets)
    }

    /// make sure this is called, otherwise things will explode
    pub fn close(self) -> (usize, usize) {
        for _ in 0..self.num_workers {
            self.workers
                .recv()
                .expect("worker not received when closing");
            // wait for all workers to complete
        }

        for worker in self.worker_map.iter() {
            let stream = { worker.lock().unwrap().stream.clone() };
            let _ = stream.synchronize();
        }

        // make sure we drop the unified buffers here:

        Arc::try_unwrap(self.worker_map)
            .ok()
            .expect("we need to free the workers on the main thread");

        let total_dropped_packets = self.drop_counter.load(Relaxed);

        let total_emitted_packets = self.counters.iter().map(|c| c.load(Relaxed)).sum::<usize>();

        (total_emitted_packets, total_dropped_packets)
    }
}
