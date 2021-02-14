use capsule::{dpdk, Mbuf, PortQueue};
use std::{
    collections::HashSet,
    error::Error,
    ffi::c_void,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

#[link(name = "kernel", kind = "static")]
extern "C" {
    fn p4_process(
        pkts: *const *mut u8,
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

fn process_once(
    port: &PortQueue,
    temp_buf: &mut Vec<*mut ()>,
    mbufs: &mut Vec<Mbuf>,
    pktbufs: &mut Vec<*mut u8>,
    lengths_in: &mut Vec<u64>,
    lengths_out: &mut Vec<u64>,
    offsets_out: &mut Vec<u64>,
    buf_size: u16,
) -> u16 {
    temp_buf.clear();
    mbufs.clear();
    lengths_in.clear();
    lengths_out.clear();
    offsets_out.clear();

    port.receive_into(temp_buf, mbufs, buf_size);

    let pkt_count = mbufs.len();

    lengths_out.reserve(pkt_count);
    offsets_out.reserve(pkt_count);

    for (i, pkt) in mbufs.into_iter().enumerate() {
        let pkt_slice = get_mbuf_data(pkt);
        pktbufs.insert(i, pkt_slice.as_mut_ptr());
        lengths_in.insert(i, pkt_slice.len() as u64);

        unsafe {
            p4_process(
                pktbufs.as_ptr(),
                lengths_in.as_ptr(),
                lengths_out.as_mut_ptr(),
                offsets_out.as_mut_ptr(),
                pkt_count as u64,
                0,
                i as u64,
            );

            // lengths_out[i] is filled by p4_process
            lengths_out.set_len(i + 1);
            offsets_out.set_len(i + 1);
        }

        if lengths_out[i] < lengths_in[i] {
            // packet is smaller:
            // in:  [HHHHDDDDD]
            // out: [0HHHDDDDD]

            let _ = pkt.shrink(0, offsets_out[i] as usize);
        } else if lengths_out[i] > lengths_in[i] {
            // packet is migger
            // in:  [HHHHDDDDD]
            // out: [HHHHDDDDD]D
            // (I hope this is fine)
            let _ = pkt.extend(
                lengths_in[i] as usize,
                (lengths_out[i] - lengths_in[i]) as usize,
            );
        }
    }

    println!("processed packets, transmitting");

    port.transmit_from(mbufs);

    pkt_count as u16
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
            args: Some("rx_pcap=dump.pcap,tx_iface=lo".to_owned()),
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

    let queue = ports.first().unwrap().queues().values().next().unwrap();

    println!("receiving packets");

    let start_t = Instant::now();
    let mut total_pkts = 0;

    let mut temp_buf = Vec::new();
    let mut mbufs = Vec::new();
    let mut pktbufs = Vec::new();
    let mut lengths_in = Vec::new();
    let mut lengths_out = Vec::new();
    let mut offsets_out = Vec::new();

    loop {
        let n_pkts = process_once(
            queue,
            &mut temp_buf,
            &mut mbufs,
            &mut pktbufs,
            &mut lengths_in,
            &mut lengths_out,
            &mut offsets_out,
            packet_buf_size,
        );
        total_pkts += n_pkts;

        if n_pkts < packet_buf_size {
            break;
        }
    }

    let end_t = Instant::now();

    println!(
        "done, took: {:?}, {} packets",
        end_t.duration_since(start_t),
        total_pkts
    );

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
