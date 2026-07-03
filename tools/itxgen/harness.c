// Differential harness: NEON 4-column kernels vs scalar int64 reference
// (both generated from the same Go source; the scalar side uses the original
// unrewritten coefficients in int64, so this validates the coefficient
// rewriting and the int32-lane range proof independently).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core_ref.h"

void goav1_idct32_col4(int32_t *base, long strideBytes, long min, long max);
void goav1_idct64_col4(int32_t *base, long strideBytes, long min, long max, int32_t *scratch);

static uint64_t s = 0x9e3779b97f4a7c15ULL;
static uint64_t rnd(void) {
  s ^= s << 13; s ^= s >> 7; s ^= s << 17;
  return s;
}

static int32_t clampv(int64_t v, int64_t mn, int64_t mx) {
  return (int32_t)(v < mn ? mn : (v > mx ? mx : v));
}

static const int32_t edges[] = {0, 1, -1, 7, -7, -32768, 32767, -131072, 131071,
                                -524288, 524287, -8388608, 8388607};

static int32_t g_scratch[512 + 64];

static void idct32_wrap(int32_t *b, long st, long mn, long mx) { goav1_idct32_col4(b, st, mn, mx); }
static void idct64_wrap(int32_t *b, long st, long mn, long mx) { goav1_idct64_col4(b, st, mn, mx, g_scratch + (rnd() % 64)); }

static int run(int n, void (*neon)(int32_t *, long, long, long),
               void (*ref)(int64_t *, int64_t, int64_t)) {
  // clamp ladder: bd 8/10/12 row+col bounds
  static const int bits[] = {16, 16, 18, 16, 20, 18};
  const int strideEls = 8; // 4 target columns + 4 padding
  for (int ci = 0; ci < 6; ci++) {
    int64_t mx = (1LL << (bits[ci] - 1)) - 1, mn = -(1LL << (bits[ci] - 1));
    for (int iter = 0; iter < 30000; iter++) {
      int32_t buf[64 * 8], want[64 * 8];
      for (int i = 0; i < n * strideEls; i++) {
        int32_t raw = (iter & 1) ? edges[rnd() % 13]
                                 : (int32_t)(rnd()) >> (rnd() % 13);
        buf[i] = clampv(raw, mn, mx);
      }
      memcpy(want, buf, sizeof(int32_t) * n * strideEls);
      // scalar reference per column
      for (int col = 0; col < 4; col++) {
        int64_t v[64];
        for (int i = 0; i < n; i++) v[i] = want[i * strideEls + col];
        ref(v, mn, mx);
        for (int i = 0; i < n; i++) want[i * strideEls + col] = (int32_t)v[i];
      }
      neon(buf, strideEls * 4, mn, mx);
      if (memcmp(buf, want, sizeof(int32_t) * n * strideEls)) {
        printf("MISMATCH n=%d clamp=%d iter=%d\n", n, bits[ci], iter);
        for (int i = 0; i < n; i++)
          for (int col = 0; col < 4; col++)
            if (buf[i * strideEls + col] != want[i * strideEls + col])
              printf("  row=%d col=%d got=%d want=%d\n", i, col,
                     buf[i * strideEls + col], want[i * strideEls + col]);
        return 1;
      }
    }
  }
  printf("dct%d col4 ok\n", n);
  return 0;
}



void goav1_idct32_row4(int32_t *r0, int32_t *r1, int32_t *r2, int32_t *r3, long min, long max);
void goav1_idct64_row4(int32_t *r0, int32_t *r1, int32_t *r2, int32_t *r3, long min, long max, int32_t *scratch);
void goav1_iadst16_col4(int32_t *base, long strideBytes, long min, long max);
void goav1_iadst16_row4(int32_t *r0, int32_t *r1, int32_t *r2, int32_t *r3, long min, long max);
static void iadst16_col4_wrap(int32_t *b, long st, long mn, long mx) { goav1_iadst16_col4(b, st, mn, mx); }

static int run_iadst16_row4(void) {
  static const int bits[] = {16, 16, 18, 16, 20, 18};
  for (int ci = 0; ci < 6; ci++) {
    int64_t mx = (1LL << (bits[ci] - 1)) - 1, mn = -(1LL << (bits[ci] - 1));
    for (int iter = 0; iter < 30000; iter++) {
      int32_t rows[4][64], want[4][64];
      for (int r = 0; r < 4; r++)
        for (int i = 0; i < 16; i++) {
          int32_t raw = (iter & 1) ? edges[rnd() % 13] : (int32_t)(rnd()) >> (rnd() % 13);
          rows[r][i] = clampv(raw, mn, mx);
        }
      memcpy(want, rows, sizeof rows);
      for (int r = 0; r < 4; r++) {
        int64_t v[64];
        for (int i = 0; i < 16; i++) v[i] = want[r][i];
        iadst16_ref(v, mn, mx);
        for (int i = 0; i < 16; i++) want[r][i] = (int32_t)v[i];
      }
      goav1_iadst16_row4(rows[0], rows[1], rows[2], rows[3], mn, mx);
      if (memcmp(rows, want, sizeof rows)) {
        printf("IADST16 ROW MISMATCH clamp=%d iter=%d\n", bits[ci], iter);
        return 1;
      }
    }
  }
  printf("iadst16 row4 ok\n");
  return 0;
}

static int run_row4(int n) {
  static const int bits[] = {16, 16, 18, 16, 20, 18};
  for (int ci = 0; ci < 6; ci++) {
    int64_t mx = (1LL << (bits[ci] - 1)) - 1, mn = -(1LL << (bits[ci] - 1));
    for (int iter = 0; iter < 30000; iter++) {
      int32_t rows[4][64], want[4][64];
      for (int r = 0; r < 4; r++)
        for (int i = 0; i < n; i++) {
          int32_t raw = (iter & 1) ? edges[rnd() % 13] : (int32_t)(rnd()) >> (rnd() % 13);
          rows[r][i] = clampv(raw, mn, mx);
        }
      memcpy(want, rows, sizeof rows);
      for (int r = 0; r < 4; r++) {
        int64_t v[64];
        for (int i = 0; i < n; i++) v[i] = want[r][i];
        if (n == 32) idct32_ref(v, mn, mx); else idct64_ref(v, mn, mx);
        for (int i = 0; i < n; i++) want[r][i] = (int32_t)v[i];
      }
      if (n == 32)
        goav1_idct32_row4(rows[0], rows[1], rows[2], rows[3], mn, mx);
      else
        goav1_idct64_row4(rows[0], rows[1], rows[2], rows[3], mn, mx, g_scratch);
      if (memcmp(rows, want, sizeof rows)) {
        printf("ROW MISMATCH n=%d clamp=%d iter=%d\n", n, bits[ci], iter);
        return 1;
      }
    }
  }
  printf("dct%d row4 ok\n", n);
  return 0;
}

int main(void) {
  if (run(32, idct32_wrap, idct32_ref)) return 1;
  if (run(64, idct64_wrap, idct64_ref)) return 1;
  if (run_row4(32)) return 1;
  if (run_row4(64)) return 1;
  if (run(16, iadst16_col4_wrap, iadst16_ref)) return 1;
  if (run_iadst16_row4()) return 1;
  return 0;
}
