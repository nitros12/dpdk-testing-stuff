use capsule::{dpdk, Mbuf, PortQueue};
use howlong::HighResolutionTimer;
use std::{
    collections::HashSet,
    error::Error,
    ffi::c_void,
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

#[repr(C)]
enum Action {
    Emit = 1,
    Drop = 0,
}

#[repr(C)]
struct Metadata {
    output_port: u32,
    output_action: Action,
    packet_length: u32,
    input_port: u32,
}

#[link(name = "kernel", kind = "static")]
extern "C" {
    fn p4_process(
        pkts: *const *mut u8,
        meta: *mut Metadata,
        lengths: *const u64,
        lengths_out: *mut u64,
        out_offsets: *mut u64,
        pkt_count: u64,
        port: u64,
        i: u64,
    );
}

fn get_mbuf_data(buf: &dpdk::Mbuf) -> &mut [u8] {
    unsafe {
        let ptr = buf.read_data_slice::<u8>(0, buf.data_len()).unwrap();
        &mut *ptr.as_ptr()
    }
}

struct PortInfo<'a> {
    port: &'a PortQueue,
    transmit_queue: Vec<Mbuf>,
    transmitted: usize,
}

impl<'a> PortInfo<'a> {
    fn new(port: &'a PortQueue) -> Self {
        Self {
            port,
            transmit_queue: Vec::new(),
            transmitted: 0,
        }
    }

    fn add_to_queue(&mut self, mbuf: Mbuf) {
        self.transmit_queue.push(mbuf);
    }

    fn flush(&mut self) {
        if !self.transmit_queue.is_empty() {
            self.transmitted += self.transmit_queue.len();

            self.port.transmit_from(&mut self.transmit_queue);
        }
    }
}

struct ProcessData<'a> {
    ports: Vec<PortInfo<'a>>,
    dropped: usize,
    drop_queue: Vec<Mbuf>,
    temp_buf: Vec<*mut ()>,
    mbufs: Vec<Mbuf>,
    pktbufs: Vec<*mut u8>,
    metas: Vec<Metadata>,
    lengths_in: Vec<u64>,
    lengths_out: Vec<u64>,
    offsets_out: Vec<u64>,
    buf_size: u16,
}

impl<'a> ProcessData<'a> {
    fn new(ports: &[&'a PortQueue], buf_size: u16) -> Self {
        let ports = ports.into_iter().cloned().map(PortInfo::new).collect();

        Self {
            ports,
            drop_queue: Vec::new(),
            dropped: 0,
            temp_buf: Vec::with_capacity(buf_size as usize),
            mbufs: Vec::with_capacity(buf_size as usize),
            pktbufs: Vec::with_capacity(buf_size as usize),
            metas: Vec::with_capacity(buf_size as usize),
            lengths_in: Vec::with_capacity(buf_size as usize),
            lengths_out: Vec::with_capacity(buf_size as usize),
            offsets_out: Vec::with_capacity(buf_size as usize),
            buf_size,
        }
    }

    fn clear(&mut self) {
        self.temp_buf.clear();
        self.mbufs.clear();
        self.metas.clear();
        self.lengths_in.clear();
        self.lengths_out.clear();
        self.offsets_out.clear();
    }

    fn flush(&mut self) {
        for port in &mut self.ports {
            port.flush();
        }

        if !self.drop_queue.is_empty() {
            self.dropped += self.drop_queue.len();
            Mbuf::free_bulk(std::mem::replace(&mut self.drop_queue, Vec::new()));
        }
    }

    fn print_stats(&self) {
        println!("dropped: {}", self.dropped);

        for (idx, port) in self.ports.iter().enumerate() {
            println!("port {}: {}", idx, port.transmitted);
        }
    }
}

fn process_once(
    port: &PortQueue,
    port_idx: u32,
    process_data: &mut ProcessData<'_>,
) -> (usize, Duration) {
    process_data.clear();

    port.receive_into(
        &mut process_data.temp_buf,
        &mut process_data.mbufs,
        process_data.buf_size,
    );

    let start = HighResolutionTimer::new();

    let pkt_count = process_data.mbufs.len();

    process_data.lengths_out.reserve(pkt_count);
    process_data.offsets_out.reserve(pkt_count);

    for (i, mut pkt) in process_data.mbufs.drain(..).enumerate() {
        let pkt_slice = get_mbuf_data(&pkt);
        process_data.pktbufs.insert(i, pkt_slice.as_mut_ptr());
        process_data.lengths_in.insert(i, pkt_slice.len() as u64);
        process_data.metas.insert(
            i,
            Metadata {
                output_port: port_idx,
                output_action: Action::Drop,
                packet_length: pkt_slice.len() as u32,
                input_port: port_idx,
            },
        );

        unsafe {
            p4_process(
                process_data.pktbufs.as_ptr(),
                process_data.metas.as_mut_ptr(),
                process_data.lengths_in.as_ptr(),
                process_data.lengths_out.as_mut_ptr(),
                process_data.offsets_out.as_mut_ptr(),
                pkt_count as u64,
                port_idx as u64,
                i as u64,
            );

            // lengths_out[i] is filled by p4_process
            process_data.lengths_out.set_len(i + 1);
            process_data.offsets_out.set_len(i + 1);
        }

        if process_data.lengths_out[i] < process_data.lengths_in[i] {
            // packet is smaller:
            // in:  [HHHHDDDDD]
            // out: [0HHHDDDDD]

            let _ = pkt.shrink(0, process_data.offsets_out[i] as usize);
        } else if process_data.lengths_out[i] > process_data.lengths_in[i] {
            // packet is migger
            // in:  [HHHHDDDDD]
            // out: [HHHHDDDDD]D
            // (I hope this is fine)
            let _ = pkt.extend(
                process_data.lengths_in[i] as usize,
                (process_data.lengths_out[i] - process_data.lengths_in[i]) as usize,
            );
        }

        match process_data.metas[i].output_action {
            Action::Emit => {
                process_data.ports[process_data.metas[i].output_port as usize].add_to_queue(pkt);
            }
            Action::Drop => {
                process_data.drop_queue.push(pkt);
            }
        }
    }

    process_data.flush();

    let elapsed = start.elapsed();

    (pkt_count, elapsed)
}

fn main_inner() -> Result<(), Box<dyn Error>> {
    let conf = capsule::config::RuntimeConfig {
        app_name: "test".to_owned(),
        secondary: false,
        app_group: None,
        master_core: dpdk::CoreId::new(0),
        cores: vec![],
        mempool: Default::default(),
        ports: vec![capsule::config::PortConfig {
            name: "testPort0".to_owned(),
            device: "net_pcap0".to_owned(),
            args: Some("rx_pcap=dump2.pcap".to_owned()),
            cores: vec![dpdk::CoreId::new(0)],
            rxd: 128,
            txd: 128,
            promiscuous: false,
            multicast: true,
            kni: false,
        }],
        dpdk_args: None,
        duration: None,
    };

    dpdk::eal_init(conf.to_eal_args())?;

    let cores = conf.all_cores();

    let sockets = cores.iter().map(|c| c.socket_id()).collect::<HashSet<_>>();

    let mut mempools = sockets
        .iter()
        .map(|s| dpdk::Mempool::new(conf.mempool.capacity, conf.mempool.cache_size, *s))
        .collect::<Result<Vec<_>, _>>()?;

    let mut ports = conf
        .ports
        .iter()
        .map(|p| {
            dpdk::PortBuilder::new(p.name.clone(), p.device.clone())?
                .cores(&p.cores)?
                .mempools(&mut mempools)
                .rx_tx_queue_capacity(p.rxd, p.txd)?
                .finish(p.promiscuous, p.multicast, p.kni)
        })
        .collect::<Result<Vec<_>, _>>()?;

    println!("starting ports");

    for port in &mut ports {
        port.start()?;
    }

    let packet_buf_size = 512;

    let port_queues: Vec<_> = ports
        .iter()
        .map(|p| p.queues().values().next().unwrap())
        .collect();
    let queue = ports.first().unwrap().queues().values().next().unwrap();

    println!("receiving packets");

    let start_t = Instant::now();
    let mut processing_duration = Duration::default();
    let mut total_pkts = 0;

    let mut process_data = ProcessData::new(&port_queues, packet_buf_size as u16);

    loop {
        let (n_pkts, duration) = process_once(queue, 0, &mut process_data);

        total_pkts += n_pkts;
        processing_duration += duration;

        if n_pkts < packet_buf_size {
            break;
        }
    }

    let total_duration = start_t.elapsed();

    let total_secs = (processing_duration.as_nanos() as f64) / 1_000_000_000f64;

    println!(
        "done, took: {:?} total, {:?} spent in-processing, {} packets, {} pkt/s",
        total_duration,
        processing_duration,
        total_pkts,
        total_pkts as f64 / total_secs
    );

    process_data.print_stats();

    for port in &mut ports {
        port.stop();
    }

    dpdk::eal_cleanup()?;

    println!("closed dpdk");

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    main_inner()?;

    println!("Closing up");

    Ok(())
}
