#include <cstdint>

// todo: we can use deltas instead of offsets, and have the compiler try and
// pack close entries together
struct trie_entry {
  uint32_t next_if_t;
  uint32_t next_if_f;
  uint32_t dest_port;
};

__constant__ struct trie_entry entries[] = {
  [0] = (struct trie_entry){.next_if_t = 2, .next_if_f = 1, .dest_port = 0},

  [1] = (struct trie_entry){.next_if_t = 1, .next_if_f = 1, .dest_port = 1},
  [2] = (struct trie_entry){.next_if_t = 2, .next_if_f = 2, .dest_port = 2},
};

__device__ uint32_t update_match_state(uint8_t ip_segment, uint32_t state) {
  struct trie_entry c = entries[state];
  for (uint8_t i = 0; i < 8; i++) {
    state = (ip_segment >> (8 - i)) & 1 ? c.next_if_t : c.next_if_f;
  }
  return state;
}

__global__ void perform(const uint8_t **pkt, uint32_t *dest_port, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }

  uint8_t a = pkt[i][30];
  uint8_t b = pkt[i][31];
  uint8_t c = pkt[i][32];
  uint8_t d = pkt[i][33];

  uint32_t lpm_state = 0;
  lpm_state = update_match_state(a, lpm_state);
  lpm_state = update_match_state(b, lpm_state);
  lpm_state = update_match_state(c, lpm_state);
  lpm_state = update_match_state(d, lpm_state);

  dest_port[i] = entries[lpm_state].dest_port;
}
