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
                device: "0000:00:08.0".to_owned(),
                args: None,
                cores: vec![dpdk::CoreId::new(0)],
                rxd: 128,
                txd: 128,
                promiscuous: false,
                multicast: true,
                kni: false,
            },
            capsule::config::PortConfig {
                name: "testPort1".to_owned(),
                device: "0000:00:09.0".to_owned(),
                args: None,
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
    println!("Hello, world!");
}
