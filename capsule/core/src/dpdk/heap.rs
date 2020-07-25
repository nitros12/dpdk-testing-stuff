use super::SocketId;
use crate::dpdk::DpdkError;
use crate::ffi::{self, AsStr, ToCString, ToResult};
use crate::{debug, info};
use failure::{Fail, Fallible};
use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::os::raw;
use std::ptr::{self, NonNull};
use std::{
    ffi::c_void,
    sync::atomic::{AtomicUsize, Ordering},
};

pub fn create_heap(name: &str) -> Fallible<()> {
    let name = name.to_cstring();
    unsafe {
        ffi::rte_malloc_heap_create(name.as_ptr() as *const i8)
            .to_result(DpdkError::from_errno)
            .map(|_| ())
    }
}

pub fn add_heap_memory(
    name: &str,
    addr: *mut c_void,
    len: usize,
    n_pages: usize,
    page_size: usize,
) -> Fallible<()> {
    let name = name.to_cstring();
    let addr_u8 = addr as *mut u8;

    let mut iovas = (0..n_pages)
        .map(|idx| {
            unsafe {
                let cur = addr_u8.offset((idx * page_size) as isize);
                // apparently needed
                cur.write_volatile(0);
                ffi::rte_mem_virt2iova(cur as *mut c_void)
            }
        })
        .collect::<Vec<_>>();

    unsafe {
        ffi::rte_malloc_heap_memory_add(
            name.as_ptr() as *const i8,
            addr,
            len as u64,
            iovas.as_mut_ptr(),
            iovas.len() as u32,
            page_size as u64,
        )
        .to_result(DpdkError::from_errno)
        .map(|_| ())
    }
}
