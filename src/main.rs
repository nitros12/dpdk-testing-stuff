use capsule::{dpdk, Mbuf, PortQueue};
use howlong::HighResolutionTimer;
use std::{
    collections::HashSet,
    error::Error,
    ffi::c_void,
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

mod gpu;
mod meta;
mod coordinator;
mod portinfo;

use crate::meta::{Metadata, Action};
use crate::portinfo::PortInfo;


fn main_inner() -> Result<(), Box<dyn Error>> {
    let (device, _context) = gpu::init()?;

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


    let (bufs_and_sockets, mut mempools): (Vec<_>, Vec<_>) = sockets
        .into_iter()
        // .map(|s| dpdk::Mempool::new(conf.mempool.capacity, conf.mempool.cache_size, s))
        .map(|_s| gpu::allocate_gpu_mempool(conf.mempool.capacity, conf.mempool.cache_size))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();

    let (bufs, sockets): (Vec<_>, Vec<_>) = bufs_and_sockets.into_iter().unzip();

    let extra_heap_mappings = cores
        .iter()
        .map(|c| c.socket_id())
        .zip(sockets.iter().cloned())
        .collect::<Vec<_>>();

    let mut ports = conf
        .ports
        .iter()
        .map(|p| {
            dpdk::PortBuilder::new(p.name.clone(), p.device.clone())?
                .cores(&p.cores)?
                .mempools(&mut mempools, &extra_heap_mappings)
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

    let start_t = Instant::now();
    let mut total_pkts = 0;

    println!("making coordinator");

    let coordinator = coordinator::Coordinator::new(&port_queues, 512, 8)?;

    println!("receiving packets");

    loop {
        let n_pkts = coordinator.process_packets(queue, 0)?;

        total_pkts += n_pkts;

        if n_pkts < packet_buf_size {
            break;
        }
    }

    coordinator.close();

    let total_duration = start_t.elapsed();

    let total_secs = (total_duration.as_nanos() as f64) / 1_000_000_000f64;

    println!(
        "done, took: {:?} total, {} packets, {} pkt/s",
        total_duration,
        total_pkts,
        total_pkts as f64 / total_secs
    );

    for port in &mut ports {
        port.stop();
    }

    dpdk::eal_cleanup()?;

    println!("closed dpdk");

    // sys exit here so that dpdk doesn't try to free our memory pools and fail,
    // we can probably do that a proper way, but this is easier
    std::process::exit(0);
}

fn main() -> Result<(), Box<dyn Error>> {
    main_inner()?;

    println!("Closing up");

    Ok(())
}
