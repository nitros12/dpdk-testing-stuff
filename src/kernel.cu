#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <arpa/inet.h>
#include <endian.h>
#include <byteswap.h>
#include <string.h>
__device__ uint64_t p4_htonll(uint64_t in) {
#if __BYTE_ORDER == __LITTLE_ENDIAN
  return ((in >> 56) & 0xffull) | ((in >> 40) & 0xff00ull)
          | ((in >> 24) & 0xff0000ull)
          | ((in >> 8) & 0xff000000ull)
          | ((in & 0xff000000ull) << 8)
          | ((in & 0xff0000ull) << 24)
          | ((in & 0xff00ull) << 40)
          | ((in & 0xffull) << 56);
#else
  return in;
#endif
}
__device__ uint32_t p4_htonl(uint32_t in) {
#if __BYTE_ORDER == __LITTLE_ENDIAN
  return ((in >> 24) & 0xff) | ((in >> 8) & 0xff00)
          | ((in & 0xff00) << 8) | ((in & 0xff) << 24);
#else
  return in;
#endif
}
__device__ uint16_t p4_htons(uint16_t in) {
#if __BYTE_ORDER == __LITTLE_ENDIAN
  return ((in >> 8) & 0xff) | ((in & 0xff) << 8);
#else
  return in;
#endif
}
__device__ void p4_memmove(void *dst, void *src, size_t n) {
  char *d = (char *)dst;
  const char *s = (const char *)src;
  if ((uintptr_t)dst < (uintptr_t)src) {
    for (size_t i = 0; i < n; i++)
      d[i] = s[i];
  } else {
    for (size_t i = n; i > 0; i--)
      d[i - 1] = s[i - 1];
  }
}

struct IPv4_h
{ uint32_t dstAddr;
  uint32_t srcAddr;
  uint16_t hdrChecksum;
  uint8_t protocol;
  uint8_t ttl;
  uint16_t fragOffset;
  uint8_t flags;
  uint16_t identification;
  uint16_t totalLen;
  uint8_t diffserv;
  uint8_t ihl;
  uint8_t version;
  _Bool p4_valid;
};
struct Ethernet_h
{ uint16_t etherType;
  uint64_t srcAddr;
  uint64_t dstAddr;
  _Bool p4_valid;
};
struct Headers_t
{ struct IPv4_h ipv4;
  struct Ethernet_h ethernet;
};
enum gpu_action {EMIT = (1) , DROP = (0)};
struct standard_metadata
{ uint32_t output_port;
  enum gpu_action output_action;
  uint32_t packet_length;
  uint32_t input_port;
};
struct test_tbl_0
{ uint8_t action_run;
  _Bool miss;
  _Bool hit;
};
struct metadata
{ char unused;
};
enum parser_states_prs {parser_state_prs_parse_ipv4 = (3) , parser_state_prs_start = (2) , parser_state_prs_accept = (1) , parser_state_prs_reject = (0)};
struct packet
{ uint8_t * pkt;
  uint16_t base;
  uint16_t end;
  uint16_t offset;
};
struct ppacket
{ struct packet * ppkt;
};
struct arg_table_test_tbl_0_allow
{ char unused;
};
struct arg_table_test_tbl_0_deny
{ char unused;
};
struct arg_table_test_tbl_0_NoAction_0
{ char unused;
};
union test_tbl_0_param_union
{ struct arg_table_test_tbl_0_NoAction_0 arg_table_test_tbl_0_NoAction_0;
  struct arg_table_test_tbl_0_deny arg_table_test_tbl_0_deny;
  struct arg_table_test_tbl_0_allow arg_table_test_tbl_0_allow;
};
struct match_tree_node
{ int16_t offsets[(16)];
  uint16_t action_idx;
  uint16_t param_idx;
};
__constant__ const union test_tbl_0_param_union arg_table_test_tbl_0_entries[] = {{.arg_table_test_tbl_0_allow = {.unused = (0)}}, {.arg_table_test_tbl_0_deny = {.unused = (0)}}};
__constant__ const struct match_tree_node test_tbl_0_search_trie[(46)] = {{.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (1), .param_idx = (0)}, {.offsets = {(42), (42), (42), (42), (42), (42), (42), (42), (42), (42), (42), (42), (42), (-1), (42), (42)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(40), (40), (40), (40), (40), (40), (-1), (40), (40), (40), (40), (40), (40), (40), (40), (40)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(37), (37), (37), (37), (37), (37), (37), (-1), (37), (37), (37), (37), (37), (37), (37), (37)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(33), (33), (33), (33), (33), (-1), (33), (33), (33), (33), (33), (33), (33), (33), (33), (33)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(28), (28), (28), (-1), (28), (28), (28), (28), (28), (28), (28), (28), (28), (28), (28), (28)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(22), (22), (22), (22), (22), (22), (22), (22), (22), (22), (-1), (22), (22), (22), (22), (22)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(15), (15), (15), (15), (15), (-1), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (2), .param_idx = (1)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (2), .param_idx = (1)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (2), .param_idx = (1)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (2), .param_idx = (1)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (2), .param_idx = (1)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (2), .param_idx = (1)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (2), .param_idx = (1)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (2), .param_idx = (1)}, {.offsets = {(-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-37), (-29), (-29)}, .action_idx = (0), .param_idx = (0)}};

__device__ static _Bool extract_packet_Ethernet_h(struct ppacket ppkt, struct Ethernet_h * hdr) {
  if ((((ppkt).ppkt)->end) < ((((ppkt).ppkt)->offset) + (14))) {
    {return false;}
  };
  uint8_t tmp_var_7 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (0)]);
  ((*(hdr)).dstAddr) = ((((*(hdr)).dstAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (40)))) | ((((tmp_var_7) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (40)));
  uint8_t tmp_var_8 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (1)]);
  ((*(hdr)).dstAddr) = ((((*(hdr)).dstAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (32)))) | ((((tmp_var_8) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (32)));
  uint8_t tmp_var_9 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (2)]);
  ((*(hdr)).dstAddr) = ((((*(hdr)).dstAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (24)))) | ((((tmp_var_9) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (24)));
  uint8_t tmp_var_10 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (3)]);
  ((*(hdr)).dstAddr) = ((((*(hdr)).dstAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (16)))) | ((((tmp_var_10) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (16)));
  uint8_t tmp_var_11 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (4)]);
  ((*(hdr)).dstAddr) = ((((*(hdr)).dstAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (8)))) | ((((tmp_var_11) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (8)));
  uint8_t tmp_var_12 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (5)]);
  ((*(hdr)).dstAddr) = ((((*(hdr)).dstAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (0)))) | ((((tmp_var_12) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (0)));
  uint8_t tmp_var_13 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (6)]);
  ((*(hdr)).srcAddr) = ((((*(hdr)).srcAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (40)))) | ((((tmp_var_13) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (40)));
  uint8_t tmp_var_14 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (7)]);
  ((*(hdr)).srcAddr) = ((((*(hdr)).srcAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (32)))) | ((((tmp_var_14) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (32)));
  uint8_t tmp_var_15 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (8)]);
  ((*(hdr)).srcAddr) = ((((*(hdr)).srcAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (24)))) | ((((tmp_var_15) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (24)));
  uint8_t tmp_var_16 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (9)]);
  ((*(hdr)).srcAddr) = ((((*(hdr)).srcAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (16)))) | ((((tmp_var_16) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (16)));
  uint8_t tmp_var_17 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (10)]);
  ((*(hdr)).srcAddr) = ((((*(hdr)).srcAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (8)))) | ((((tmp_var_17) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (8)));
  uint8_t tmp_var_18 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (11)]);
  ((*(hdr)).srcAddr) = ((((*(hdr)).srcAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (0)))) | ((((tmp_var_18) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (0)));
  uint8_t tmp_var_19 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (12)]);
  ((*(hdr)).etherType) = ((((*(hdr)).etherType) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (8)))) | ((((tmp_var_19) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (8)));
  uint8_t tmp_var_20 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (13)]);
  ((*(hdr)).etherType) = ((((*(hdr)).etherType) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (0)))) | ((((tmp_var_20) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (0)));
  (((ppkt).ppkt)->offset) += (14);
  ((hdr)->p4_valid) = (true);
  return true;
}

__device__ static _Bool extract_packet_IPv4_h(struct ppacket ppkt, struct IPv4_h * hdr) {
  if ((((ppkt).ppkt)->end) < ((((ppkt).ppkt)->offset) + (20))) {
    {return false;}
  };
  uint8_t tmp_var_26 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (0)]);
  ((*(hdr)).version) = ((((*(hdr)).version) & (~(((((uint64_t)(1)) << (4)) - ((uint64_t)(1))) << (0)))) | ((((tmp_var_26) >> (4)) & ((((uint64_t)(1)) << (4)) - ((uint64_t)(1)))) << (0)));
  ((*(hdr)).ihl) = ((((*(hdr)).ihl) & (~(((((uint64_t)(1)) << (4)) - ((uint64_t)(1))) << (0)))) | ((((tmp_var_26) >> (0)) & ((((uint64_t)(1)) << (4)) - ((uint64_t)(1)))) << (0)));
  uint8_t tmp_var_27 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (1)]);
  ((*(hdr)).diffserv) = ((((*(hdr)).diffserv) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (0)))) | ((((tmp_var_27) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (0)));
  uint8_t tmp_var_28 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (2)]);
  ((*(hdr)).totalLen) = ((((*(hdr)).totalLen) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (8)))) | ((((tmp_var_28) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (8)));
  uint8_t tmp_var_29 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (3)]);
  ((*(hdr)).totalLen) = ((((*(hdr)).totalLen) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (0)))) | ((((tmp_var_29) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (0)));
  uint8_t tmp_var_30 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (4)]);
  ((*(hdr)).identification) = ((((*(hdr)).identification) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (8)))) | ((((tmp_var_30) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (8)));
  uint8_t tmp_var_31 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (5)]);
  ((*(hdr)).identification) = ((((*(hdr)).identification) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (0)))) | ((((tmp_var_31) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (0)));
  uint8_t tmp_var_32 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (6)]);
  ((*(hdr)).flags) = ((((*(hdr)).flags) & (~(((((uint64_t)(1)) << (3)) - ((uint64_t)(1))) << (0)))) | ((((tmp_var_32) >> (5)) & ((((uint64_t)(1)) << (3)) - ((uint64_t)(1)))) << (0)));
  ((*(hdr)).fragOffset) = ((((*(hdr)).fragOffset) & (~(((((uint64_t)(1)) << (5)) - ((uint64_t)(1))) << (8)))) | ((((tmp_var_32) >> (0)) & ((((uint64_t)(1)) << (5)) - ((uint64_t)(1)))) << (8)));
  uint8_t tmp_var_33 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (7)]);
  ((*(hdr)).fragOffset) = ((((*(hdr)).fragOffset) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (0)))) | ((((tmp_var_33) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (0)));
  uint8_t tmp_var_34 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (8)]);
  ((*(hdr)).ttl) = ((((*(hdr)).ttl) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (0)))) | ((((tmp_var_34) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (0)));
  uint8_t tmp_var_35 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (9)]);
  ((*(hdr)).protocol) = ((((*(hdr)).protocol) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (0)))) | ((((tmp_var_35) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (0)));
  uint8_t tmp_var_36 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (10)]);
  ((*(hdr)).hdrChecksum) = ((((*(hdr)).hdrChecksum) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (8)))) | ((((tmp_var_36) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (8)));
  uint8_t tmp_var_37 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (11)]);
  ((*(hdr)).hdrChecksum) = ((((*(hdr)).hdrChecksum) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (0)))) | ((((tmp_var_37) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (0)));
  uint8_t tmp_var_38 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (12)]);
  ((*(hdr)).srcAddr) = ((((*(hdr)).srcAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (24)))) | ((((tmp_var_38) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (24)));
  uint8_t tmp_var_39 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (13)]);
  ((*(hdr)).srcAddr) = ((((*(hdr)).srcAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (16)))) | ((((tmp_var_39) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (16)));
  uint8_t tmp_var_40 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (14)]);
  ((*(hdr)).srcAddr) = ((((*(hdr)).srcAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (8)))) | ((((tmp_var_40) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (8)));
  uint8_t tmp_var_41 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (15)]);
  ((*(hdr)).srcAddr) = ((((*(hdr)).srcAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (0)))) | ((((tmp_var_41) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (0)));
  uint8_t tmp_var_42 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (16)]);
  ((*(hdr)).dstAddr) = ((((*(hdr)).dstAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (24)))) | ((((tmp_var_42) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (24)));
  uint8_t tmp_var_43 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (17)]);
  ((*(hdr)).dstAddr) = ((((*(hdr)).dstAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (16)))) | ((((tmp_var_43) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (16)));
  uint8_t tmp_var_44 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (18)]);
  ((*(hdr)).dstAddr) = ((((*(hdr)).dstAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (8)))) | ((((tmp_var_44) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (8)));
  uint8_t tmp_var_45 = ((((ppkt).ppkt)->pkt)[(((ppkt).ppkt)->offset) + (19)]);
  ((*(hdr)).dstAddr) = ((((*(hdr)).dstAddr) & (~(((((uint64_t)(1)) << (8)) - ((uint64_t)(1))) << (0)))) | ((((tmp_var_45) >> (0)) & ((((uint64_t)(1)) << (8)) - ((uint64_t)(1)))) << (0)));
  (((ppkt).ppkt)->offset) += (20);
  ((hdr)->p4_valid) = (true);
  return true;
}

__device__ static _Bool prs(struct ppacket packet, struct Headers_t * hdr, struct metadata * meta, struct standard_metadata * std_meta) {
  enum parser_states_prs tmp_var_5 = (parser_state_prs_start);
  for (;;) {
    {switch (tmp_var_5) {
       {case (parser_state_prs_start): {struct ppacket tmp_var_21 = (packet);
                                        struct Ethernet_h * tmp_var_22 = (&((*(hdr)).ethernet));
                                        struct Ethernet_h tmp_var_23 = (*(tmp_var_22));
                                        _Bool tmp_var_24 = ((extract_packet_Ethernet_h)((tmp_var_21), (&(tmp_var_23))));
                                        (*(tmp_var_22)) = (tmp_var_23);
                                        if (!(tmp_var_24)) {
                                          {(tmp_var_5) = (parser_state_prs_reject);
                                           break;}
                                        };
                                        0;
                                        enum parser_states_prs tmp_var_25;
                                        switch (((*(hdr)).ethernet).etherType) {
                                          {case (2048): {(tmp_var_25) = (parser_state_prs_parse_ipv4);
                                                         break;};
                                           default: {(tmp_var_25) = (parser_state_prs_accept);
                                                     break;};}
                                        };
                                        (tmp_var_5) = (tmp_var_25);
                                        break;};
        case (parser_state_prs_parse_ipv4): {struct ppacket tmp_var_46 = (packet);
                                             struct IPv4_h * tmp_var_47 = (&((*(hdr)).ipv4));
                                             struct IPv4_h tmp_var_48 = (*(tmp_var_47));
                                             _Bool tmp_var_49 = ((extract_packet_IPv4_h)((tmp_var_46), (&(tmp_var_48))));
                                             (*(tmp_var_47)) = (tmp_var_48);
                                             if (!(tmp_var_49)) {
                                               {(tmp_var_5) = (parser_state_prs_reject);
                                                break;}
                                             };
                                             0;
                                             (tmp_var_5) = (parser_state_prs_accept);
                                             break;};
        case (parser_state_prs_accept): {return true;};
        case (parser_state_prs_reject): {return false;};}
     };}
  };
}

__device__ static void allow(struct standard_metadata * std_meta) {
  ((*(std_meta)).output_action) = (EMIT);
}

__device__ static void deny(struct standard_metadata * std_meta) {
  ((*(std_meta)).output_action) = (DROP);
}

__device__ static void NoAction_0(void) {
  0;
}

__device__ static const struct match_tree_node * table_trie_driver_w8(const struct match_tree_node * node, uint32_t value) {
  size_t idx;
  for ((idx) = (1); (idx) <= (8); ++(idx)) {
    {(node) += (((node)->offsets)[((value) >> ((32) - ((idx) * (4)))) & (((1) << (4)) - (1))]);}
  };
  return node;
}

__device__ static void pipe(struct Headers_t * hdr, struct metadata * meta, struct standard_metadata * std_meta) {
  uint32_t tmp_var_53 = (((*(hdr)).ipv4).dstAddr);
  const struct match_tree_node * tmp_var_54 = (&((test_tbl_0_search_trie)[45]));
  (tmp_var_54) = ((table_trie_driver_w8)((tmp_var_54), (tmp_var_53)));
  _Bool tmp_var_61;
  switch ((tmp_var_54)->action_idx) {
    {default: {(tmp_var_61) = (false);
               break;};
     case (1): {(tmp_var_61) = (true);
                struct standard_metadata * tmp_var_56 = (&(*(std_meta)));
                struct standard_metadata tmp_var_57 = (*(tmp_var_56));
                (allow)((&(tmp_var_57)));
                (*(tmp_var_56)) = (tmp_var_57);
                0;
                break;};
     case (0): {(tmp_var_61) = (false);
                (NoAction_0)();
                0;
                break;};
     case (2): {(tmp_var_61) = (true);
                struct standard_metadata * tmp_var_59 = (&(*(std_meta)));
                struct standard_metadata tmp_var_60 = (*(tmp_var_59));
                (deny)((&(tmp_var_60)));
                (*(tmp_var_59)) = (tmp_var_60);
                0;
                break;};}
  };
  (struct test_tbl_0){.action_run = ((tmp_var_54)->action_idx), .miss = (!(tmp_var_61)), .hit = (tmp_var_61)};
}

__device__ static void write_partial(uint8_t * addr, uint8_t width, uint8_t shift, uint8_t value) {
  ((*(addr)) = ((*(addr)) & (~(((((uint8_t)(1)) << (width)) - ((uint8_t)(1))) << (shift))))) | ((value) << (shift));
}

__device__ static void emit_packet_Ethernet_h(struct ppacket ppkt, struct Ethernet_h * value) {
  uint64_t tmp_var_64 = (0);
  uint64_t tmp_var_65 = ((p4_htonll)((((*(value)).dstAddr) << (16))));
  uint8_t * tmp_var_66 = ((uint8_t *)(&(tmp_var_65)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_64) / (8))) + (0))) = ((tmp_var_66)[0]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_64) / (8))) + (1))) = ((tmp_var_66)[1]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_64) / (8))) + (2))) = ((tmp_var_66)[2]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_64) / (8))) + (3))) = ((tmp_var_66)[3]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_64) / (8))) + (4))) = ((tmp_var_66)[4]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_64) / (8))) + (5))) = ((tmp_var_66)[5]);
  (tmp_var_64) += (48);
  uint64_t tmp_var_67 = ((p4_htonll)((((*(value)).srcAddr) << (16))));
  uint8_t * tmp_var_68 = ((uint8_t *)(&(tmp_var_67)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_64) / (8))) + (0))) = ((tmp_var_68)[0]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_64) / (8))) + (1))) = ((tmp_var_68)[1]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_64) / (8))) + (2))) = ((tmp_var_68)[2]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_64) / (8))) + (3))) = ((tmp_var_68)[3]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_64) / (8))) + (4))) = ((tmp_var_68)[4]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_64) / (8))) + (5))) = ((tmp_var_68)[5]);
  (tmp_var_64) += (48);
  uint16_t tmp_var_69 = ((p4_htons)((((*(value)).etherType) << (0))));
  uint8_t * tmp_var_70 = ((uint8_t *)(&(tmp_var_69)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_64) / (8))) + (0))) = ((tmp_var_70)[0]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_64) / (8))) + (1))) = ((tmp_var_70)[1]);
  (tmp_var_64) += (16);
  (((ppkt).ppkt)->offset) = ((tmp_var_64) / (8));
}

__device__ static void emit_packet_IPv4_h(struct ppacket ppkt, struct IPv4_h * value) {
  uint64_t tmp_var_74 = (0);
  uint8_t tmp_var_75 = ((*(value)).version);
  uint8_t * tmp_var_76 = ((uint8_t *)(&(tmp_var_75)));
  (write_partial)((&(*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (0)))), (4), (4), (((tmp_var_76)[0]) >> (0)));
  (tmp_var_74) += (4);
  uint8_t tmp_var_77 = ((*(value)).ihl);
  uint8_t * tmp_var_78 = ((uint8_t *)(&(tmp_var_77)));
  (write_partial)((&(*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (0)))), (4), (0), (((tmp_var_78)[0]) >> (0)));
  (tmp_var_74) += (4);
  uint8_t tmp_var_79 = ((*(value)).diffserv);
  uint8_t * tmp_var_80 = ((uint8_t *)(&(tmp_var_79)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (0))) = ((tmp_var_80)[0]);
  (tmp_var_74) += (8);
  uint16_t tmp_var_81 = ((p4_htons)((((*(value)).totalLen) << (0))));
  uint8_t * tmp_var_82 = ((uint8_t *)(&(tmp_var_81)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (0))) = ((tmp_var_82)[0]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (1))) = ((tmp_var_82)[1]);
  (tmp_var_74) += (16);
  uint16_t tmp_var_83 = ((p4_htons)((((*(value)).identification) << (0))));
  uint8_t * tmp_var_84 = ((uint8_t *)(&(tmp_var_83)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (0))) = ((tmp_var_84)[0]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (1))) = ((tmp_var_84)[1]);
  (tmp_var_74) += (16);
  uint8_t tmp_var_85 = ((*(value)).flags);
  uint8_t * tmp_var_86 = ((uint8_t *)(&(tmp_var_85)));
  (write_partial)((&(*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (0)))), (3), (5), (((tmp_var_86)[0]) >> (0)));
  (tmp_var_74) += (3);
  uint16_t tmp_var_87 = ((p4_htons)((((*(value)).fragOffset) << (3))));
  uint8_t * tmp_var_88 = ((uint8_t *)(&(tmp_var_87)));
  (write_partial)((&(*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (0)))), (5), (0), (((tmp_var_88)[0]) >> (3)));
  (write_partial)(((&(*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (0)))) + (1)), (3), (5), ((tmp_var_88)[0]));
  (write_partial)((&(*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (1)))), (5), (3), (((tmp_var_88)[1]) >> (3)));
  (tmp_var_74) += (13);
  uint8_t tmp_var_89 = ((*(value)).ttl);
  uint8_t * tmp_var_90 = ((uint8_t *)(&(tmp_var_89)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (0))) = ((tmp_var_90)[0]);
  (tmp_var_74) += (8);
  uint8_t tmp_var_91 = ((*(value)).protocol);
  uint8_t * tmp_var_92 = ((uint8_t *)(&(tmp_var_91)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (0))) = ((tmp_var_92)[0]);
  (tmp_var_74) += (8);
  uint16_t tmp_var_93 = ((p4_htons)((((*(value)).hdrChecksum) << (0))));
  uint8_t * tmp_var_94 = ((uint8_t *)(&(tmp_var_93)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (0))) = ((tmp_var_94)[0]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (1))) = ((tmp_var_94)[1]);
  (tmp_var_74) += (16);
  uint32_t tmp_var_95 = ((p4_htonl)((((*(value)).srcAddr) << (0))));
  uint8_t * tmp_var_96 = ((uint8_t *)(&(tmp_var_95)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (0))) = ((tmp_var_96)[0]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (1))) = ((tmp_var_96)[1]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (2))) = ((tmp_var_96)[2]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (3))) = ((tmp_var_96)[3]);
  (tmp_var_74) += (32);
  uint32_t tmp_var_97 = ((p4_htonl)((((*(value)).dstAddr) << (0))));
  uint8_t * tmp_var_98 = ((uint8_t *)(&(tmp_var_97)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (0))) = ((tmp_var_98)[0]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (1))) = ((tmp_var_98)[1]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (2))) = ((tmp_var_98)[2]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_74) / (8))) + (3))) = ((tmp_var_98)[3]);
  (tmp_var_74) += (32);
  (((ppkt).ppkt)->offset) = ((tmp_var_74) / (8));
}

__device__ static void adjust_packet(struct packet * pkt, int current_size, int final_size) {
  ((pkt)->offset) = (0);
  if ((current_size) == (final_size)) {
    {return;}
  };
  if ((final_size) < (current_size)) {
    {((pkt)->base) = ((current_size) - (final_size));
     ((pkt)->offset) = ((current_size) - (final_size));}
  } else {
    {(p4_memmove)((((pkt)->pkt) + (final_size)), (((pkt)->pkt) + (current_size)), (((pkt)->end) - (current_size)));
     ((pkt)->end) += ((final_size) - (current_size));}
  };
}

__device__ static void dprs(struct ppacket packet, struct Headers_t hdr) {
  (adjust_packet)(((packet).ppkt), (272), (272));
  if (((hdr).ethernet).p4_valid) {
    {struct ppacket tmp_var_71 = (packet);
     struct Ethernet_h * tmp_var_72 = (&((hdr).ethernet));
     struct Ethernet_h tmp_var_73 = (*(tmp_var_72));
     (emit_packet_Ethernet_h)((tmp_var_71), (&(tmp_var_73)));
     (*(tmp_var_72)) = (tmp_var_73);
     0;}
  };
  if (((hdr).ipv4).p4_valid) {
    {struct ppacket tmp_var_99 = (packet);
     struct IPv4_h * tmp_var_100 = (&((hdr).ipv4));
     struct IPv4_h tmp_var_101 = (*(tmp_var_100));
     (emit_packet_IPv4_h)((tmp_var_99), (&(tmp_var_101)));
     (*(tmp_var_100)) = (tmp_var_101);
     0;}
  };
}

extern "C" __global__ void p4_process(uint8_t ** pkts, struct standard_metadata * std_meta, uint64_t * lengths, uint64_t * out_lengths, uint64_t * out_offsets, uint64_t pkt_count, uint64_t port) {
  uint64_t i = ((((blockIdx).x) * ((blockDim).x)) + ((threadIdx).x));
  if ((i) >= (pkt_count)) {
    {return;}
  };
  struct metadata meta = ((struct metadata){(0)});
  struct packet pkt = ((struct packet){.pkt = ((pkts)[i]), .end = ((lengths)[i]), .base = (0), .offset = (0)});
  struct Headers_t hdr = ((struct Headers_t){(0)});
  struct ppacket ppkt = ((struct ppacket){.ppkt = (&(pkt))});
  (prs)((ppkt), (&(hdr)), (&(meta)), (&((std_meta)[i])));
  (pipe)((&(hdr)), (&(meta)), (&((std_meta)[i])));
  (dprs)((ppkt), (hdr));
  ((out_lengths)[i]) = ((pkt).end);
  ((out_offsets)[i]) = ((pkt).base);
}

