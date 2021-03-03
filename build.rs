use std::{env, process::Command};

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    Command::new("nvcc")
        .args(&["-ccbin", "clang", "-ptx", "src/kernel.cu", "-o"])
        .arg(&format!("{}/kernel.ptx", out_dir))
        .status()
        .unwrap();

    //println!("cargo:rerun-if-changed=src/kernel.cu");
}
