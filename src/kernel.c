#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <arpa/inet.h>
#include <endian.h>
#include <byteswap.h>
#include <string.h>
uint64_t p4_htonll(uint64_t in) {
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
uint32_t p4_htonl(uint32_t in) {
#if __BYTE_ORDER == __LITTLE_ENDIAN
  return ((in >> 24) & 0xff) | ((in >> 8) & 0xff00)
          | ((in & 0xff00) << 8) | ((in & 0xff) << 24);
#else
  return in;
#endif
}
uint16_t p4_htons(uint16_t in) {
#if __BYTE_ORDER == __LITTLE_ENDIAN
  return ((in >> 8) & 0xff) | ((in & 0xff) << 8);
#else
  return in;
#endif
}
void p4_memmove(void *dst, void *src, size_t n) {
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
enum ubpf_action {REDIRECT , PASS , DROP , ABORT};
struct standard_metadata
{ uint32_t clone_port;
  _Bool clone;
  uint32_t output_port;
  enum ubpf_action output_action;
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
enum parser_states_prs {parser_state_prs_parse_ipv4 , parser_state_prs_start , parser_state_prs_accept , parser_state_prs_reject};
struct packet
{ uint8_t * pkt;
  uint16_t base;
  uint16_t end;
  uint16_t offset;
};
struct ppacket
{ struct packet * ppkt;
};
struct arg_table_test_tbl_0_mod_nw_tos
{ uint32_t out_port;
};
struct arg_table_test_tbl_0_NoAction_0
{ char unused;
};
union test_tbl_0_param_union
{ struct arg_table_test_tbl_0_NoAction_0 arg_table_test_tbl_0_NoAction_0;
  struct arg_table_test_tbl_0_mod_nw_tos arg_table_test_tbl_0_mod_nw_tos;
};
struct match_tree_node
{ int16_t offsets[(16)];
  uint16_t action_idx;
  uint16_t param_idx;
};
const union test_tbl_0_param_union arg_table_test_tbl_0_entries[] = {{.arg_table_test_tbl_0_mod_nw_tos = {.out_port = (1)}}, {.arg_table_test_tbl_0_mod_nw_tos = {.out_port = (0)}}, {.arg_table_test_tbl_0_mod_nw_tos = {.out_port = (2)}}};
const struct match_tree_node test_tbl_0_search_trie[(47)] = {{.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (1), .param_idx = (0)}, {.offsets = {(-1), (7), (43), (43), (43), (43), (43), (43), (43), (43), (43), (43), (43), (43), (43), (43)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (41), (41), (41), (41), (41), (41), (41), (41), (41), (41), (41), (41), (41), (41), (41)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (38), (38), (38), (38), (38), (38), (38), (38), (38), (38), (38), (38), (38), (38), (38)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (34), (34), (34), (34), (34), (34), (34), (34), (34), (34), (34), (34), (34), (34), (34)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (29), (29), (29), (29), (29), (29), (29), (29), (29), (29), (29), (29), (29), (29), (29)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (23), (23), (23), (23), (23), (23), (23), (23), (23), (23), (23), (23), (23), (23), (23)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (1), .param_idx = (1)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (1), .param_idx = (2)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (1), .param_idx = (2)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (1), .param_idx = (2)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (1), .param_idx = (2)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (1), .param_idx = (2)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (1), .param_idx = (2)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (1), .param_idx = (2)}, {.offsets = {(-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1)}, .action_idx = (0), .param_idx = (0)}, {.offsets = {(0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0), (0)}, .action_idx = (1), .param_idx = (2)}, {.offsets = {(-38), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29)}, .action_idx = (0), .param_idx = (0)}};

static _Bool extract_packet_Ethernet_h(struct ppacket ppkt, struct Ethernet_h * hdr) {
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

static _Bool extract_packet_IPv4_h(struct ppacket ppkt, struct IPv4_h * hdr) {
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

static _Bool prs(struct ppacket packet, struct Headers_t * hdr, struct metadata * meta, struct standard_metadata * std_meta) {
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

static void mod_nw_tos(uint32_t out_port, uint8_t * i0_0, struct standard_metadata * std_meta, struct Headers_t * hdr) {
  (((*(hdr)).ipv4).diffserv) = ((*(i0_0)) + (1));
  ((*(std_meta)).output_action) = (REDIRECT);
  ((*(std_meta)).output_port) = (out_port);
  (*(i0_0)) = ((*(i0_0)) + (1));
}

static void NoAction_0(void) {
  0;
}

static const struct match_tree_node * table_trie_driver_w8(const struct match_tree_node * node, uint32_t value) {
  size_t idx;
  for ((idx) = (0); (idx) < (8); ++(idx)) {
    {(node) += (((node)->offsets)[((value) >> ((32) - ((idx) * (4)))) & (((1) << (4)) - (1))]);}
  };
  return node;
}

static void pipe(struct Headers_t * hdr, struct metadata * meta, struct standard_metadata * std_meta) {
  uint8_t i0_0;
  (i0_0) = (0);
  uint32_t tmp_var_54 = ((*(std_meta)).input_port);
  const struct match_tree_node * tmp_var_55 = (&((test_tbl_0_search_trie)[46]));
  (tmp_var_55) = ((table_trie_driver_w8)((tmp_var_55), (tmp_var_54)));
  _Bool tmp_var_69;
  switch ((tmp_var_55)->action_idx) {
    {default: {(tmp_var_69) = (false);};
     case (1): {(tmp_var_69) = (true);
                uint32_t tmp_var_56 = ((((arg_table_test_tbl_0_entries)[(tmp_var_55)->param_idx]).arg_table_test_tbl_0_mod_nw_tos).out_port);
                uint32_t tmp_var_62 = (tmp_var_56);
                uint8_t * tmp_var_63 = (&(i0_0));
                uint8_t tmp_var_64 = (*(tmp_var_63));
                struct standard_metadata * tmp_var_65 = (&(*(std_meta)));
                struct standard_metadata tmp_var_66 = (*(tmp_var_65));
                struct Headers_t * tmp_var_67 = (&(*(hdr)));
                struct Headers_t tmp_var_68 = (*(tmp_var_67));
                (mod_nw_tos)((tmp_var_62), (&(tmp_var_64)), (&(tmp_var_66)), (&(tmp_var_68)));
                (*(tmp_var_63)) = (tmp_var_64);
                (*(tmp_var_65)) = (tmp_var_66);
                (*(tmp_var_67)) = (tmp_var_68);
                0;
                break;};
     case (0): {(tmp_var_69) = (false);
                (NoAction_0)();
                0;
                break;};}
  };
  (struct test_tbl_0){.action_run = ((tmp_var_55)->action_idx), .miss = (!(tmp_var_69)), .hit = (tmp_var_69)};
}

static void write_partial(uint8_t * addr, uint8_t width, uint8_t shift, uint8_t value) {
  ((*(addr)) = ((*(addr)) & (~(((((uint8_t)(1)) << (width)) - ((uint8_t)(1))) << (shift))))) | ((value) << (shift));
}

static void emit_packet_Ethernet_h(struct ppacket ppkt, struct Ethernet_h * value) {
  uint64_t tmp_var_72 = (0);
  uint64_t tmp_var_73 = ((p4_htonll)((((*(value)).dstAddr) << (16))));
  uint8_t * tmp_var_74 = ((uint8_t *)(&(tmp_var_73)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_72) / (8))) + (0))) = ((tmp_var_74)[0]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_72) / (8))) + (1))) = ((tmp_var_74)[1]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_72) / (8))) + (2))) = ((tmp_var_74)[2]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_72) / (8))) + (3))) = ((tmp_var_74)[3]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_72) / (8))) + (4))) = ((tmp_var_74)[4]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_72) / (8))) + (5))) = ((tmp_var_74)[5]);
  (tmp_var_72) += (48);
  uint64_t tmp_var_75 = ((p4_htonll)((((*(value)).srcAddr) << (16))));
  uint8_t * tmp_var_76 = ((uint8_t *)(&(tmp_var_75)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_72) / (8))) + (0))) = ((tmp_var_76)[0]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_72) / (8))) + (1))) = ((tmp_var_76)[1]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_72) / (8))) + (2))) = ((tmp_var_76)[2]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_72) / (8))) + (3))) = ((tmp_var_76)[3]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_72) / (8))) + (4))) = ((tmp_var_76)[4]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_72) / (8))) + (5))) = ((tmp_var_76)[5]);
  (tmp_var_72) += (48);
  uint16_t tmp_var_77 = ((p4_htons)((((*(value)).etherType) << (0))));
  uint8_t * tmp_var_78 = ((uint8_t *)(&(tmp_var_77)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_72) / (8))) + (0))) = ((tmp_var_78)[0]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_72) / (8))) + (1))) = ((tmp_var_78)[1]);
  (tmp_var_72) += (16);
  (((ppkt).ppkt)->offset) = ((tmp_var_72) / (8));
}

static void emit_packet_IPv4_h(struct ppacket ppkt, struct IPv4_h * value) {
  uint64_t tmp_var_82 = (0);
  uint8_t tmp_var_83 = ((*(value)).version);
  uint8_t * tmp_var_84 = ((uint8_t *)(&(tmp_var_83)));
  (write_partial)((&(*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (0)))), (4), (4), (((tmp_var_84)[0]) >> (0)));
  (tmp_var_82) += (4);
  uint8_t tmp_var_85 = ((*(value)).ihl);
  uint8_t * tmp_var_86 = ((uint8_t *)(&(tmp_var_85)));
  (write_partial)((&(*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (0)))), (4), (0), (((tmp_var_86)[0]) >> (0)));
  (tmp_var_82) += (4);
  uint8_t tmp_var_87 = ((*(value)).diffserv);
  uint8_t * tmp_var_88 = ((uint8_t *)(&(tmp_var_87)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (0))) = ((tmp_var_88)[0]);
  (tmp_var_82) += (8);
  uint16_t tmp_var_89 = ((p4_htons)((((*(value)).totalLen) << (0))));
  uint8_t * tmp_var_90 = ((uint8_t *)(&(tmp_var_89)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (0))) = ((tmp_var_90)[0]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (1))) = ((tmp_var_90)[1]);
  (tmp_var_82) += (16);
  uint16_t tmp_var_91 = ((p4_htons)((((*(value)).identification) << (0))));
  uint8_t * tmp_var_92 = ((uint8_t *)(&(tmp_var_91)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (0))) = ((tmp_var_92)[0]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (1))) = ((tmp_var_92)[1]);
  (tmp_var_82) += (16);
  uint8_t tmp_var_93 = ((*(value)).flags);
  uint8_t * tmp_var_94 = ((uint8_t *)(&(tmp_var_93)));
  (write_partial)((&(*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (0)))), (3), (5), (((tmp_var_94)[0]) >> (0)));
  (tmp_var_82) += (3);
  uint16_t tmp_var_95 = ((p4_htons)((((*(value)).fragOffset) << (3))));
  uint8_t * tmp_var_96 = ((uint8_t *)(&(tmp_var_95)));
  (write_partial)((&(*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (0)))), (5), (0), (((tmp_var_96)[0]) >> (3)));
  (write_partial)(((&(*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (0)))) + (1)), (3), (5), ((tmp_var_96)[0]));
  (write_partial)((&(*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (1)))), (5), (3), (((tmp_var_96)[1]) >> (3)));
  (tmp_var_82) += (13);
  uint8_t tmp_var_97 = ((*(value)).ttl);
  uint8_t * tmp_var_98 = ((uint8_t *)(&(tmp_var_97)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (0))) = ((tmp_var_98)[0]);
  (tmp_var_82) += (8);
  uint8_t tmp_var_99 = ((*(value)).protocol);
  uint8_t * tmp_var_100 = ((uint8_t *)(&(tmp_var_99)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (0))) = ((tmp_var_100)[0]);
  (tmp_var_82) += (8);
  uint16_t tmp_var_101 = ((p4_htons)((((*(value)).hdrChecksum) << (0))));
  uint8_t * tmp_var_102 = ((uint8_t *)(&(tmp_var_101)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (0))) = ((tmp_var_102)[0]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (1))) = ((tmp_var_102)[1]);
  (tmp_var_82) += (16);
  uint32_t tmp_var_103 = ((p4_htonl)((((*(value)).srcAddr) << (0))));
  uint8_t * tmp_var_104 = ((uint8_t *)(&(tmp_var_103)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (0))) = ((tmp_var_104)[0]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (1))) = ((tmp_var_104)[1]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (2))) = ((tmp_var_104)[2]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (3))) = ((tmp_var_104)[3]);
  (tmp_var_82) += (32);
  uint32_t tmp_var_105 = ((p4_htonl)((((*(value)).dstAddr) << (0))));
  uint8_t * tmp_var_106 = ((uint8_t *)(&(tmp_var_105)));
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (0))) = ((tmp_var_106)[0]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (1))) = ((tmp_var_106)[1]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (2))) = ((tmp_var_106)[2]);
  (*(((((ppkt).ppkt)->pkt) + ((tmp_var_82) / (8))) + (3))) = ((tmp_var_106)[3]);
  (tmp_var_82) += (32);
  (((ppkt).ppkt)->offset) = ((tmp_var_82) / (8));
}

static void adjust_packet(struct packet * pkt, int current_size, int final_size) {
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

static void dprs(struct ppacket packet, struct Headers_t hdr) {
  (adjust_packet)(((packet).ppkt), (272), (272));
  if (((hdr).ethernet).p4_valid) {
    {struct ppacket tmp_var_79 = (packet);
     struct Ethernet_h * tmp_var_80 = (&((hdr).ethernet));
     struct Ethernet_h tmp_var_81 = (*(tmp_var_80));
     (emit_packet_Ethernet_h)((tmp_var_79), (&(tmp_var_81)));
     (*(tmp_var_80)) = (tmp_var_81);
     0;}
  };
  if (((hdr).ipv4).p4_valid) {
    {struct ppacket tmp_var_107 = (packet);
     struct IPv4_h * tmp_var_108 = (&((hdr).ipv4));
     struct IPv4_h tmp_var_109 = (*(tmp_var_108));
     (emit_packet_IPv4_h)((tmp_var_107), (&(tmp_var_109)));
     (*(tmp_var_108)) = (tmp_var_109);
     0;}
  };
}

void p4_process(uint8_t ** pkts, uint64_t * lengths, uint64_t * out_lengths, uint64_t * out_offsets, uint64_t pkt_count, uint64_t port, uint64_t i) {
  if ((i) >= (pkt_count)) {
    {return;}
  };
  struct standard_metadata std_meta = ((struct standard_metadata){.input_port = (port), .packet_length = ((lengths)[i])});
  struct packet pkt = ((struct packet){.pkt = ((pkts)[i]), .end = ((lengths)[i]), .base = (0), .offset = (0)});
  struct Headers_t hdr = ((struct Headers_t){(0)});
  struct ppacket ppkt = ((struct ppacket){.ppkt = (&(pkt))});
  (prs)((ppkt), (&(hdr)), (NULL), (&(std_meta)));
  (pipe)((&(hdr)), (NULL), (&(std_meta)));
  (dprs)((ppkt), (hdr));
  ((out_lengths)[i]) = ((pkt).end);
  ((out_offsets)[i]) = ((pkt).base);
}

