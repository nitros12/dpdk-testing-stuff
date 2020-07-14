use capsule::dpdk;

fn main() {
    let conf = capsule::config::RuntimeConfig {
        app_name: "test".to_owned(),
        secondary: false,
        app_group: None,
        master_core: dpdk::CoreId::new(0),
        cores: vec![],
        mempool: Default::default(),
        ports: vec![
            capsule::config::PortConfig {
                name: "testPort0".to_owned(),
                device: "net_pcap0".to_owned(),
                args: Some("rx_pcap=input.pcap,tx_iface=lo".to_owned()),
                cores: vec![dpdk::CoreId::new(0)],
                rxd: 128,
                txd: 128,
                promiscuous: false,
                multicast: true,
                kni: false,
            },
        ],
        dpdk_args: None,
        duration: None,
    };


    dpdk::eal_init(dbg!(conf.to_eal_args())).unwrap();

    let src = r#"
        __kernel void add(__global float* buffer, float scalar) {
            buffer[get_global_id(0)] += scalar;
        }
    "#;

    let pro_que = ocl::ProQue::builder()
        .src(src)
        .dims(1 << 20)
        .build()
        .unwrap();

    let buf = pro_que.create_buffer::<f32>()
        .unwrap();

    let kernel = pro_que.kernel_builder("add")
                        .arg(&buf)
                        .arg(10f32)
                        .build().unwrap();

    unsafe { kernel.enq().unwrap(); }

    println!("Hello, world!");
}
