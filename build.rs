fn main() {
    cc::Build::new()
        .opt_level(3)
        .file("src/kernel.c")
        .compile("kernel")
}
