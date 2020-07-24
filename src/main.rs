use capsule::{dpdk, runtime::CoreMapBuilder};
use context::Context;
use device::Device;
use rustacuda::{
    context::{self, ContextFlags},
    device,
    memory::{DeviceBuffer, UnifiedBuffer},
    module::Module,
    stream::{self, StreamFlags},
    CudaFlags, launch,
};
use std::{collections::HashSet, error::Error, ffi::{CString, c_void}, env};
use stream::Stream;

fn get_mbuf_data(buf: &mut dpdk::Mbuf) -> &mut [u8] {
    unsafe {
        let ptr = buf.read_data_slice::<u8>(0, buf.data_len()).unwrap();
        &mut *ptr.as_ptr()
    }
}

fn allocate_gpu_mempool(
    capacity: usize,
    cache_size: usize,
    socket_id: dpdk::SocketId,
) -> Result<capsule::dpdk::Mempool, Box<dyn Error>> {
    let mut buf = UnifiedBuffer::new(&0u8, capacity)?;
    let pool = dpdk::Mempool::new_external(
        &[buf.as_mut_ptr() as *mut c_void],
        capacity,
        cache_size,
        socket_id,
    )?;
    Ok(pool)
}

fn main() -> Result<(), Box<dyn Error>> {
    rustacuda::init(CudaFlags::empty())?;

    let device = Device::get_device(0)?;

    let context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let ptx = CString::new(include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx")))?;
    let module = Module::load_from_string(&ptx)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

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

    dpdk::eal_init(conf.to_eal_args())?;

    let cores = conf.all_cores();

    let sockets = cores.iter().map(|c| c.socket_id()).collect::<HashSet<_>>();

    let mut mempools = sockets
        .into_iter()
        // .map(|s| dpdk::Mempool::new(conf.mempool.capacity, conf.mempool.cache_size, s))
        .map(|s| allocate_gpu_mempool(conf.mempool.capacity, conf.mempool.cache_size, s))
        .collect::<Result<Vec<_>, _>>()?;

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
            dpdk::PortBuilder::new(p.name.clone(), p.device.clone())?
                .cores(&p.cores)?
                .mempools(&mut mempools)
                .rx_tx_queue_capacity(p.rxd, p.txd)?
                .finish(p.promiscuous, p.multicast, p.kni)
        })
        .collect::<Result<Vec<_>, _>>()?;

    for port in &mut ports {
        port.start()?;
    }

    let queue = ports.first().unwrap().queues().values().next().unwrap();

    // can we remove this allocation
    let mut mbufs = queue.receive();
    let pkts = mbufs.iter_mut().map(|m| get_mbuf_data(m).as_mut_ptr() as usize).collect::<Vec<_>>();
    let mut output = UnifiedBuffer::new(&0u32, pkts.len())?;
    let mut pkts_dev = unsafe { DeviceBuffer::from_slice_async(&pkts, &stream) }?;

    unsafe {
        launch!(module.perform<<<pkts.len() as u32, 1, 0, stream>>>(
            pkts_dev.as_device_ptr(),
            output.as_unified_ptr(),
            pkts.len()
        ))?;
    }
   
    stream.synchronize()?;

    println!("Hello, world!");

    dpdk::eal_cleanup()?;

    Ok(())
}
