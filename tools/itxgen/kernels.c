// 4-column inverse DCT32/DCT64 NEON kernels for goav1.
//
// Ported from dav1d's high-bitdepth arm64 inverse transforms
// (src/arm/64/itx16.S: inv_dct_4s_x16_neon / inv_dct64_step1+2_neon shape):
// four adjacent columns ride the four int32 lanes of each .4s register, the
// butterfly multiplies are plain mul/mla/mls .4s with srshr rounding, and the
// stage clamps are smax+smin. Coefficients with |w|>2048 are rewritten as
// w-4096 plus a compensating post-shift add of the input (exact integer
// identity), which together with the +/-2^19 stage clamp bound keeps every
// int32 lane free of overflow (machine-checked in gen.py).
#include "core_neon.h"

void goav1_idct32_col4(int32_t *base, long strideBytes, long min, long max) {
  int32x4_t mn = vdupq_n_s32((int32_t)min);
  int32x4_t mx = vdupq_n_s32((int32_t)max);
  int32x4_t v[32];
  const char *p = (const char *)base;
#pragma clang loop unroll(full)
  for (int i = 0; i < 32; i++)
    v[i] = vld1q_s32((const int32_t *)(p + (long)i * strideBytes));
  idct32_core(v, mn, mx);
  char *q = (char *)base;
#pragma clang loop unroll(full)
  for (int i = 0; i < 32; i++)
    vst1q_s32((int32_t *)(q + (long)i * strideBytes), v[i]);
}

// The 64-point kernel keeps its stage buffer in caller-provided memory (bf,
// 64 int32x4_t) and runs each stage as independent 2-slot butterfly groups,
// so the compiled stack frame stays within Go's NOSPLIT budget. dav1d's
// dct64 likewise works through a stack buffer (itx16.S inv_dct64_step1/2).
void goav1_idct64_col4(int32_t *base, long strideBytes, long min, long max,
                       int32_t *scratch) {
  int32x4_t mn = vdupq_n_s32((int32_t)min);
  int32x4_t mx = vdupq_n_s32((int32_t)max);
  idct64_mem((char *)base, strideBytes, (int32x4_t *)scratch, mn, mx);
}

// --- inverse ADST16 (flip handled in the Go adapter by row-reversal) ---

void goav1_iadst16_col4(int32_t *base, long strideBytes, long min, long max) {
  int32x4_t mn = vdupq_n_s32((int32_t)min);
  int32x4_t mx = vdupq_n_s32((int32_t)max);
  int32x4_t v[16];
  const char *p = (const char *)base;
#pragma clang loop unroll(full)
  for (int i = 0; i < 16; i++)
    v[i] = vld1q_s32((const int32_t *)(p + (long)i * strideBytes));
  iadst16_core(v, mn, mx);
  char *q = (char *)base;
#pragma clang loop unroll(full)
  for (int i = 0; i < 16; i++)
    vst1q_s32((int32_t *)(q + (long)i * strideBytes), v[i]);
}

// --- four-row kernels: transpose 4x4 blocks so four rows ride the lanes ---

static inline void tr4(int32x4_t a, int32x4_t b, int32x4_t c, int32x4_t d,
                       int32x4_t *o0, int32x4_t *o1, int32x4_t *o2, int32x4_t *o3) {
  int32x4x2_t p0 = vtrnq_s32(a, b);
  int32x4x2_t p1 = vtrnq_s32(c, d);
  *o0 = vcombine_s32(vget_low_s32(p0.val[0]), vget_low_s32(p1.val[0]));
  *o1 = vcombine_s32(vget_low_s32(p0.val[1]), vget_low_s32(p1.val[1]));
  *o2 = vcombine_s32(vget_high_s32(p0.val[0]), vget_high_s32(p1.val[0]));
  *o3 = vcombine_s32(vget_high_s32(p0.val[1]), vget_high_s32(p1.val[1]));
}

void goav1_idct8_row4(int32_t *r0, int32_t *r1, int32_t *r2, int32_t *r3,
                      long min, long max) {
  int32x4_t mn = vdupq_n_s32((int32_t)min);
  int32x4_t mx = vdupq_n_s32((int32_t)max);
  int32x4_t v[8];
#pragma clang loop unroll(full)
  for (int i = 0; i < 2; i++)
    tr4(vld1q_s32(r0 + 4 * i), vld1q_s32(r1 + 4 * i),
        vld1q_s32(r2 + 4 * i), vld1q_s32(r3 + 4 * i),
        &v[4 * i], &v[4 * i + 1], &v[4 * i + 2], &v[4 * i + 3]);
  idct8_core(v, mn, mx);
#pragma clang loop unroll(full)
  for (int i = 0; i < 2; i++) {
    int32x4_t a, b, c, d;
    tr4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3], &a, &b, &c, &d);
    vst1q_s32(r0 + 4 * i, a);
    vst1q_s32(r1 + 4 * i, b);
    vst1q_s32(r2 + 4 * i, c);
    vst1q_s32(r3 + 4 * i, d);
  }
}

void goav1_idct16_row4(int32_t *r0, int32_t *r1, int32_t *r2, int32_t *r3,
                       long min, long max) {
  int32x4_t mn = vdupq_n_s32((int32_t)min);
  int32x4_t mx = vdupq_n_s32((int32_t)max);
  int32x4_t v[16];
#pragma clang loop unroll(full)
  for (int i = 0; i < 4; i++)
    tr4(vld1q_s32(r0 + 4 * i), vld1q_s32(r1 + 4 * i),
        vld1q_s32(r2 + 4 * i), vld1q_s32(r3 + 4 * i),
        &v[4 * i], &v[4 * i + 1], &v[4 * i + 2], &v[4 * i + 3]);
  idct16_core(v, mn, mx);
#pragma clang loop unroll(full)
  for (int i = 0; i < 4; i++) {
    int32x4_t a, b, c, d;
    tr4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3], &a, &b, &c, &d);
    vst1q_s32(r0 + 4 * i, a);
    vst1q_s32(r1 + 4 * i, b);
    vst1q_s32(r2 + 4 * i, c);
    vst1q_s32(r3 + 4 * i, d);
  }
}

void goav1_idct32_row4(int32_t *r0, int32_t *r1, int32_t *r2, int32_t *r3,
                       long min, long max) {
  int32x4_t mn = vdupq_n_s32((int32_t)min);
  int32x4_t mx = vdupq_n_s32((int32_t)max);
  int32x4_t v[32];
#pragma clang loop unroll(full)
  for (int i = 0; i < 8; i++)
    tr4(vld1q_s32(r0 + 4 * i), vld1q_s32(r1 + 4 * i),
        vld1q_s32(r2 + 4 * i), vld1q_s32(r3 + 4 * i),
        &v[4 * i], &v[4 * i + 1], &v[4 * i + 2], &v[4 * i + 3]);
  idct32_core(v, mn, mx);
#pragma clang loop unroll(full)
  for (int i = 0; i < 8; i++) {
    int32x4_t a, b, c, d;
    tr4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3], &a, &b, &c, &d);
    vst1q_s32(r0 + 4 * i, a);
    vst1q_s32(r1 + 4 * i, b);
    vst1q_s32(r2 + 4 * i, c);
    vst1q_s32(r3 + 4 * i, d);
  }
}

void goav1_iadst16_row4(int32_t *r0, int32_t *r1, int32_t *r2, int32_t *r3,
                        long min, long max) {
  int32x4_t mn = vdupq_n_s32((int32_t)min);
  int32x4_t mx = vdupq_n_s32((int32_t)max);
  int32x4_t v[16];
#pragma clang loop unroll(full)
  for (int i = 0; i < 4; i++)
    tr4(vld1q_s32(r0 + 4 * i), vld1q_s32(r1 + 4 * i),
        vld1q_s32(r2 + 4 * i), vld1q_s32(r3 + 4 * i),
        &v[4 * i], &v[4 * i + 1], &v[4 * i + 2], &v[4 * i + 3]);
  iadst16_core(v, mn, mx);
#pragma clang loop unroll(full)
  for (int i = 0; i < 4; i++) {
    int32x4_t a, b, c, d;
    tr4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3], &a, &b, &c, &d);
    vst1q_s32(r0 + 4 * i, a);
    vst1q_s32(r1 + 4 * i, b);
    vst1q_s32(r2 + 4 * i, c);
    vst1q_s32(r3 + 4 * i, d);
  }
}

// scratch: 512 int32 (2KB): [0,256) element-major transpose buffer,
// [256,512) the 64-entry stage buffer.
void goav1_idct64_row4(int32_t *r0, int32_t *r1, int32_t *r2, int32_t *r3,
                       long min, long max, int32_t *scratch) {
  int32x4_t mn = vdupq_n_s32((int32_t)min);
  int32x4_t mx = vdupq_n_s32((int32_t)max);
  int32_t *temp = scratch;
#pragma clang loop unroll(full)
  for (int i = 0; i < 16; i++) {
    int32x4x4_t q;
    q.val[0] = vld1q_s32(r0 + 4 * i);
    q.val[1] = vld1q_s32(r1 + 4 * i);
    q.val[2] = vld1q_s32(r2 + 4 * i);
    q.val[3] = vld1q_s32(r3 + 4 * i);
    vst4q_s32(temp + 16 * i, q); // element-major: element K at temp+4K
  }
  idct64_mem((char *)temp, 16, (int32x4_t *)(scratch + 256), mn, mx);
#pragma clang loop unroll(full)
  for (int i = 0; i < 16; i++) {
    int32x4x4_t q = vld4q_s32(temp + 16 * i);
    vst1q_s32(r0 + 4 * i, q.val[0]);
    vst1q_s32(r1 + 4 * i, q.val[1]);
    vst1q_s32(r2 + 4 * i, q.val[2]);
    vst1q_s32(r3 + 4 * i, q.val[3]);
  }
}
