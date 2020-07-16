use capsule::{dpdk, runtime::CoreMapBuilder};
use std::collections::HashSet;

fn get_mbuf_data(buf: &dpdk::Mbuf) -> &[u8] {
    unsafe {
        let ptr = buf.read_data_slice::<u8>(0, buf.data_len()).unwrap();
        &*ptr.as_ptr()
    }
}

fn write_to_buffer(in_pkts: &[dpdk::Mbuf], buf: &ocl::Buffer<u8>, offsets: &ocl::Buffer<usize>) -> ocl::EventList {
    let mut offset_so_far = 0;
    let mut evt_list = ocl::EventList::new();
    for (idx, pkt) in in_pkts.iter().enumerate() {
        let data = get_mbuf_data(pkt);
        buf.write(data)
            .offset(offset_so_far)
            .enew(&mut evt_list)
            .enq()
            .unwrap();
        offsets.write(&[offset_so_far] as &[usize])
            .offset(idx)
            .enew(&mut evt_list)
            .enq()
            .unwrap();
        offset_so_far += pkt.data_len();
    }
    evt_list
}

fn main() {
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
            args: Some("rx_pcap=input.pcap,tx_iface=lo".to_owned()),
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

    dpdk::eal_init(dbg!(conf.to_eal_args())).unwrap();

    let cores = conf.all_cores();

    let sockets = cores.iter().map(|c| c.socket_id()).collect::<HashSet<_>>();

    let mut mempools = sockets
        .into_iter()
        .map(|s| dpdk::Mempool::new(conf.mempool.capacity, conf.mempool.cache_size, s).unwrap())
        .collect::<Vec<_>>();

    // let core_map = CoreMapBuilder::new()
    //     .app_name(&conf.app_name)
    //     .cores(&cores)
    //     .master_core(conf.master_core)
    //     .mempools(&mut mempools)
    //     .finish()
    //     .unwrap();

    let mut ports = conf
        .ports
        .iter()
        .map(|p| {
            dpdk::PortBuilder::new(p.name.clone(), p.device.clone())
                .unwrap()
                .cores(&p.cores)
                .unwrap()
                .mempools(&mut mempools)
                .rx_tx_queue_capacity(p.rxd, p.txd)
                .unwrap()
                .finish(p.promiscuous, p.multicast, p.kni)
                .unwrap()
        })
        .collect::<Vec<_>>();

    for port in &mut ports {
        port.start().unwrap();
    }

    let queue = ports.first().unwrap().queues().values().next().unwrap();

    let src = r#"
        __kernel void add(__global uchar* buffer, __global ulong* offsets, ulong len) {
            if (get_global_id(0) >= len) {
                return;
            }

            buffer[offsets[get_global_id(0)]] = 1;
        }
    "#;

    let pro_que = ocl::ProQue::builder()
        .src(src)
        .dims(1 << 20)
        .build()
        .unwrap();

    let buf = pro_que.create_buffer::<u8>().unwrap();
    let offsets = pro_que.create_buffer::<usize>().unwrap();

    let msg = queue.receive();

    let read_guard = write_to_buffer(&msg, &buf, &offsets);

    let mut res = vec![0; 10];
    buf.read(&mut res).enq().unwrap();
    println!("{:?}", res);

    let kernel = pro_que
        .kernel_builder("add")
        .arg(&buf)
        .arg(&offsets)
        .arg(msg.len())
        .build()
        .unwrap();

    unsafe {
        kernel.cmd().ewait(&read_guard).enq().unwrap();
    }

    let mut res = vec![0; 10];
    buf.read(&mut res).enq().unwrap();
    println!("{:?}", res);

    println!("flushing queue");

    pro_que.queue().finish().unwrap();

    println!("Hello, world!");

    dpdk::eal_cleanup().unwrap();
}
