#!/usr/bin/env python

import sys
import typing
from itertools import cycle

import click
import scapy.utils
from scapy.all import IP, UDP, Ether, Raw
from scapy.volatile import RandIP


@click.command()
@click.option(
    "-n", "--number", type=int, help="Number of packets to write", required=True
)
@click.option(
    "-l",
    "--length",
    type=click.IntRange(0, 1300),
    help="Payload length of packet",
    required=True,
)
@click.option("-o", "--output", type=click.File("wb"), required=True)
def main(number: int, length: int, output: typing.IO):
    """Generate `number` udp packets with payloads of length `length` and write them to `output`"""

    o = scapy.utils.PcapWriter(output)

    payload = scapy.utils.randstring(length)

    p0 = (
        Ether()
        / IP(dst="213.163.87.109", src=RandIP())
        / UDP(dport=123)
        / Raw(load=payload)
    ).build()
    p1 = (
        Ether()
        / IP(dst="213.163.87.110", src=RandIP())
        / UDP(dport=123)
        / Raw(load=payload)
    ).build()

    o.write_header(p0)

    pkts = cycle([p0, p1])

    for _ in range(number):
        pkt = next(pkts)
        o.write_packet(pkt)

    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main()
