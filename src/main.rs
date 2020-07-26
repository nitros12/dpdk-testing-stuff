use capsule::dpdk;
use context::Context;
use device::Device;
use rustacuda::{
    context::{self, ContextFlags},
    device, launch,
    memory::{DeviceBuffer, UnifiedBuffer},
    module::Module,
    stream::{self, StreamFlags},
    CudaFlags,
};
use std::{
    collections::HashSet,
    env,
    error::Error,
    ffi::{c_void, CString},
    sync::atomic::{AtomicUsize, Ordering},
};
use stream::Stream;

fn get_mbuf_data(buf: &mut dpdk::Mbuf) -> &mut [u8] {
    unsafe {
        let ptr = buf.read_data_slice::<u8>(0, buf.data_len()).unwrap();
        &mut *ptr.as_ptr()
    }
}

// https://github.com/DPDK/dpdk/blob/master/app/test-pmd/testpmd.c#L614
fn calc_total_pages(capacity: usize, mbuf_size: usize, page_size: usize) -> usize {
    let header_len = 128 << 20;

    let obj_size = dpdk::calc_object_size(mbuf_size as u32, 0) as usize;

    let mbuf_per_page = page_size / obj_size;
    let leftover = (capacity % mbuf_per_page) > 0;
    let n_pages = (capacity / mbuf_per_page) + leftover as usize;

    let mbuf_mem = n_pages * page_size;

    let total_mem_unaligned = mbuf_mem + header_len;
    let total_pages = (total_mem_unaligned + page_size - 1) / page_size;

    total_pages
}

fn allocate_gpu_mempool(
    capacity: usize,
    cache_size: usize,
) -> Result<((UnifiedBuffer<u8>, dpdk::SocketId), capsule::dpdk::Mempool), Box<dyn Error>> {
    static CUDA_HEAP_COUNT: AtomicUsize = AtomicUsize::new(0);
    let n = CUDA_HEAP_COUNT.fetch_add(1, Ordering::Relaxed);
    let heap_name = format!("cuda_heap{}", n);
    dpdk::create_heap(&heap_name)?;

    let mbuf_size = capsule::ffi::RTE_MBUF_DEFAULT_BUF_SIZE as usize;

    // assume we're using normal size pages for now.
    let page_size = page_size::get();
    let num_pages = calc_total_pages(capacity, mbuf_size, page_size);
    let len = num_pages * page_size;

    println!(
        "allocating unified buffers of total len: {} (num_pages: {}, page_size: {})",
        len, num_pages, page_size
    );

    let mut buf = UnifiedBuffer::new(&0u8, len)?;
    let buf_ptr = buf.as_mut_ptr() as *mut c_void;

    dpdk::add_heap_memory(&heap_name, buf_ptr, len, num_pages, page_size)?;

    let heap_socket = dpdk::SocketId::of_heap(&heap_name)?;

    let pool = dpdk::Mempool::new(capacity, cache_size, heap_socket)?;
    println!("allocated pool");
    Ok(((buf, heap_socket), pool))
}

fn main_inner() -> Result<(), Box<dyn Error>> {
    simple_logger::init().unwrap();

    rustacuda::init(CudaFlags::empty())?;

    let device = Device::get_device(0)?;

    let _context =
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

    let (bufs_and_sockets, mut mempools): (Vec<_>, Vec<_>) = sockets
        .into_iter()
        // .map(|s| dpdk::Mempool::new(conf.mempool.capacity, conf.mempool.cache_size, s))
        .map(|_s| allocate_gpu_mempool(conf.mempool.capacity, conf.mempool.cache_size))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();

    let (bufs, sockets): (Vec<_>, Vec<_>) = bufs_and_sockets.into_iter().unzip();

    // let core_map = CoreMapBuilder::new()
    //     .app_name(&conf.app_name)
    //     .cores(&cores)
    //     .master_core(conf.master_core)
    //     .mempools(&mut mempools)
    //     .finish()
    //     .unwrap();

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

    let queue = ports.first().unwrap().queues().values().next().unwrap();

    println!("receiving packets");

    // can we remove this allocation
    let mut mbufs = queue.receive();

    println!("received packets");

    let pkts = mbufs
        .iter_mut()
        .map(|m| get_mbuf_data(m).as_mut_ptr() as usize)
        .collect::<Vec<_>>();
    let mut output = UnifiedBuffer::new(&0u32, pkts.len())?;
    let mut pkts_dev = unsafe { DeviceBuffer::from_slice_async(&pkts, &stream) }?;

    println!("running kernel, pkts len: {}", pkts.len());

    unsafe {
        launch!(module.perform<<<pkts.len() as u32, 1, 0, stream>>>(
            pkts_dev.as_device_ptr(),
            output.as_unified_ptr(),
            pkts.len()
        ))?;
    }

    stream.synchronize()?;

    println!("done");

    for port in &mut ports {
        port.stop();
    }

    drop(bufs);
    drop(ports);
    drop(mempools);

    dpdk::eal_cleanup()?;

    println!("closed dpdk");

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    main_inner()?;

    println!("Closing up");

    Ok(())
}
