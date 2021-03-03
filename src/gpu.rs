use capsule::dpdk;
use rustacuda::prelude::*;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::error::Error;
use page_size;

pub fn init() -> Result<(Device, Context), Box<dyn Error>> {
    rustacuda::init(CudaFlags::empty())?;

    let device = Device::get_device(0)?;

    let context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    Ok((device, context))
}

// https://github.com/DPDK/dpdk/blob/master/app/test-pmd/testpmd.c#L614
pub fn calc_total_pages(capacity: usize, mbuf_size: usize, page_size: usize) -> usize {
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

pub fn allocate_gpu_mempool(
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

    println!("allocated unified memory, adding to dpdk");

    dpdk::add_heap_memory(&heap_name, buf_ptr, len, num_pages, page_size)?;

    let heap_socket = dpdk::SocketId::of_heap(&heap_name)?;

    let pool = dpdk::Mempool::new(capacity, cache_size, heap_socket)?;
    println!("allocated pool");
    Ok(((buf, heap_socket), pool))
}
