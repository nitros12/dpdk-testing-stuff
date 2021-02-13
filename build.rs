fn main() {
    cc::Build::new()
        .file("src/kernel.c")
        .compile("kernel")
}
