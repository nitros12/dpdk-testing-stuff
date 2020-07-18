// todo: we can use deltas instead of offsets, and have the compiler try and
// pack close entries together
struct trie_entry {
  uint next_if_t;
  uint next_if_f;
  uint dest_port;
};

__constant struct trie_entry entries[] = {
  [0] = (struct trie_entry){.next_if_t = 2, .next_if_f = 1, .dest_port = 0},

  [1] = (struct trie_entry){.next_if_t = 1, .next_if_f = 1, .dest_port = 1},
  [2] = (struct trie_entry){.next_if_t = 2, .next_if_f = 2, .dest_port = 2},
};

uint update_match_state(uchar ip_segment, uint state) {
  struct trie_entry c = entries[state];
  for (uchar i = 0; i < 8; i++) {
    state = (ip_segment >> (8 - i)) & 1 ? c.next_if_t : c.next_if_f;
  }
  return state;
}

__kernel void add(__global uchar *buffer, __global ulong *offsets, __global uint *dest_port, ulong len) {
  if (get_global_id(0) >= len) {
    return;
  }

  __global uchar *pkt = buffer + offsets[get_global_id(0)];

  uchar a = pkt[30];
  uchar b = pkt[31];
  uchar c = pkt[32];
  uchar d = pkt[33];

  uint lpm_state = 0;
  lpm_state = update_match_state(a, lpm_state);
  lpm_state = update_match_state(b, lpm_state);
  lpm_state = update_match_state(c, lpm_state);
  lpm_state = update_match_state(d, lpm_state);

  dest_port[get_global_id(0)] = entries[lpm_state].dest_port;
}
