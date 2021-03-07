use capsule::dpdk;
use std::{collections::HashSet, error::Error, time::Instant};
use structopt::StructOpt;

mod coordinator;
mod gpu;
mod meta;
mod portinfo;

#[derive(Debug, StructOpt)]
#[structopt(name = "dpdk-gpu")]
struct Opt {
    #[structopt(short, long)]
    source: String,
}

fn main_inner() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();

    let (_device, _context) = gpu::init()?;

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
            args: Some(format!("rx_pcap={}", opt.source)),
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

    let (_bufs, sockets): (Vec<_>, Vec<_>) = bufs_and_sockets.into_iter().unzip();

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

    let packet_buf_size = 16;

    let port_queues: Vec<_> = ports
        .iter()
        .map(|p| p.queues().values().next().unwrap())
        .collect();
    let queue = ports.first().unwrap().queues().values().next().unwrap();

    println!("making coordinator");

    let coordinator = coordinator::Coordinator::new(&port_queues, packet_buf_size, 8)?;

    println!("receiving packets");

    let start_t = Instant::now();

    loop {
        let n_pkts = coordinator.process_packets(queue, 0)?;

        if n_pkts < packet_buf_size {
            break;
        }

    }

    println!("huh");

    let (total_emitted_packets, total_dropped_packets) = coordinator.close();

    let total_processed_packets = total_emitted_packets + total_dropped_packets;

    let total_duration = start_t.elapsed();

    let total_secs = (total_duration.as_nanos() as f64) / 1_000_000_000f64;

    println!(
        "done, took: {:?} total, {} packets ({} emitted, {} dropped), {} pkt/s",
        total_duration,
        total_processed_packets,
        total_emitted_packets,
        total_dropped_packets,
        total_processed_packets as f64 / total_secs
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
