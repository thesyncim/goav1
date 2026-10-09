// SPDX-License-Identifier: BSD-2-Clause

#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* dav1d src/mc.h decl_mc_fn / decl_mct_fn, BITDEPTH=16 ABI. Strides are bytes. */
typedef void (*put_fn_t)(uint16_t *dst, ptrdiff_t dst_stride,
                         const uint16_t *src, ptrdiff_t src_stride,
                         int w, int h, int mx, int my, int bitdepth_max);
typedef void (*prep_fn_t)(int16_t *tmp, const uint16_t *src,
                          ptrdiff_t src_stride, int w, int h,
                          int mx, int my, int bitdepth_max);
extern void dav1d_put_8tap_regular_smooth_16bpc_neon(
    uint16_t *, ptrdiff_t, const uint16_t *, ptrdiff_t,
    int, int, int, int, int);
extern void dav1d_prep_8tap_regular_smooth_16bpc_neon(
    int16_t *, const uint16_t *, ptrdiff_t, int, int, int, int, int);

static put_fn_t put_kernel = dav1d_put_8tap_regular_smooth_16bpc_neon;
static prep_fn_t prep_kernel = dav1d_prep_8tap_regular_smooth_16bpc_neon;
static volatile uint64_t result_sink;
static const int sides[] = { 8, 16, 32, 64, 128 };
static const int ref_side = 144;
static const int pad = 8;
static const int bitdepth_max = 1023;
static const int mx = 6;
static const int my = 9;

static void die(const char *what, const char *path) {
    fprintf(stderr, "%s: %s: %s\n", what, path, strerror(errno));
    exit(2);
}

static void *read_file(const char *dir, const char *name, size_t expected) {
    char path[1024];
    if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= (int) sizeof(path)) {
        fprintf(stderr, "path too long\n");
        exit(2);
    }
    FILE *f = fopen(path, "rb");
    if (!f) die("open", path);
    void *data = malloc(expected ? expected : 1);
    if (!data) die("malloc", path);
    size_t n = fread(data, 1, expected, f);
    int extra = fgetc(f);
    if (ferror(f) || n != expected || extra != EOF) {
        fprintf(stderr, "fixture length mismatch: %s got=%zu expected=%zu\n", path, n, expected);
        exit(2);
    }
    fclose(f);
    return data;
}

static void path_name(char *buf, size_t n, const char *prefix, int side) {
    if (snprintf(buf, n, "%s_%dx%d_u16le.bin", prefix, side, side) >= (int)n) {
        fprintf(stderr, "fixture name too long\n");
        exit(2);
    }
}

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

static int verify(const char *dir) {
    const size_t ref_bytes = (size_t) ref_side * ref_side * sizeof(uint16_t);
    const uint16_t *convolve_ref = read_file(dir, "convolve_ref_144x144_u16le.bin", ref_bytes);
    const uint16_t *compound_ref = read_file(dir, "compound_ref_144x144_u16le.bin", ref_bytes);
    int failures = 0;

    for (size_t si = 0; si < sizeof(sides) / sizeof(sides[0]); si++) {
        const int side = sides[si];
        const size_t count = (size_t) side * side;
        char name[128];
        uint16_t *put_out = calloc(count, sizeof(*put_out));
        int16_t *prep_out = calloc(count, sizeof(*prep_out));
        if (!put_out || !prep_out) die("calloc", "kernel outputs");
        path_name(name, sizeof(name), "put_expected", side);
        const uint16_t *put_expected = read_file(dir, name, count * sizeof(uint16_t));
        path_name(name, sizeof(name), "prep_expected", side);
        const uint16_t *prep_expected = read_file(dir, name, count * sizeof(uint16_t));

        const uint16_t *put_src = convolve_ref + pad * ref_side + pad;
        const uint16_t *prep_src = compound_ref + pad * ref_side + pad;
        put_kernel(put_out, side * (ptrdiff_t) sizeof(uint16_t),
                   put_src, ref_side * (ptrdiff_t) sizeof(uint16_t),
                   side, side, mx, my, bitdepth_max);
        prep_kernel(prep_out, prep_src, ref_side * (ptrdiff_t) sizeof(uint16_t),
                    side, side, mx, my, bitdepth_max);

        size_t put_diffs = 0, prep_diffs = 0;
        size_t first_put = count, first_prep = count;
        for (size_t i = 0; i < count; i++) {
            if (put_out[i] != put_expected[i]) {
                if (first_put == count) first_put = i;
                put_diffs++;
            }
            /* dav1d PREP_BIAS is 8192; compare its signed sample in the same
               uint16 offset domain as Go's compound CONV_BUF sample. */
            const uint16_t prep_offset = (uint16_t) ((int32_t) prep_out[i] + 32768);
            if (prep_offset != prep_expected[i]) {
                if (first_prep == count) first_prep = i;
                prep_diffs++;
            }
        }
        fprintf(stderr, "verify,%dx%d,put_diffs=%zu,prep_diffs=%zu", side, side, put_diffs, prep_diffs);
        if (first_put < count) fprintf(stderr, ",first_put=%zu:%u/%u", first_put, put_out[first_put], put_expected[first_put]);
        if (first_prep < count) fprintf(stderr, ",first_prep=%zu:%u/%u", first_prep,
            (unsigned) ((int32_t) prep_out[first_prep] + 32768), prep_expected[first_prep]);
        fputc('\n', stderr);
        if (put_diffs || prep_diffs) failures++;

        result_sink += put_out[count - 1] + (uint16_t) prep_out[count - 1];
        free((void *) put_expected);
        free((void *) prep_expected);
        free(put_out);
        free(prep_out);
    }
    free((void *) convolve_ref);
    free((void *) compound_ref);
    fprintf(stderr, "verify summary: %s; sink=%" PRIu64 "\n", failures ? "FAIL" : "PASS", result_sink);
    return failures ? 1 : 0;
}

static uint64_t run_put(const uint16_t *src, uint16_t *out, int side, double seconds, double *elapsed) {
    const double start = now_seconds();
    const double stop = start + seconds;
    uint64_t calls = 0;
    do {
        for (int i = 0; i < 16; i++) {
            put_kernel(out, side * (ptrdiff_t) sizeof(uint16_t),
                       src, ref_side * (ptrdiff_t) sizeof(uint16_t),
                       side, side, mx, my, bitdepth_max);
        }
        calls += 16;
    } while (now_seconds() < stop);
    *elapsed = now_seconds() - start;
    result_sink += out[(size_t) side * side - 1];
    return calls;
}

static uint64_t run_prep(const uint16_t *src, int16_t *out, int side, double seconds, double *elapsed) {
    const double start = now_seconds();
    const double stop = start + seconds;
    uint64_t calls = 0;
    do {
        for (int i = 0; i < 16; i++) {
            prep_kernel(out, src, ref_side * (ptrdiff_t) sizeof(uint16_t),
                        side, side, mx, my, bitdepth_max);
        }
        calls += 16;
    } while (now_seconds() < stop);
    *elapsed = now_seconds() - start;
    result_sink += (uint16_t) out[(size_t) side * side - 1];
    return calls;
}

static int benchmark(const char *dir) {
    const size_t ref_bytes = (size_t) ref_side * ref_side * sizeof(uint16_t);
    const uint16_t *convolve_ref = read_file(dir, "convolve_ref_144x144_u16le.bin", ref_bytes);
    const uint16_t *compound_ref = read_file(dir, "compound_ref_144x144_u16le.bin", ref_bytes);
    puts("op,shape,trial,calls,elapsed_s,ns_per_call");

    for (size_t si = 0; si < sizeof(sides) / sizeof(sides[0]); si++) {
        const int side = sides[si];
        const size_t count = (size_t) side * side;
        uint16_t *put_out = calloc(count, sizeof(*put_out));
        int16_t *prep_out = calloc(count, sizeof(*prep_out));
        if (!put_out || !prep_out) die("calloc", "kernel outputs");
        const uint16_t *put_src = convolve_ref + pad * ref_side + pad;
        const uint16_t *prep_src = compound_ref + pad * ref_side + pad;

        /* One 200ms warmup, then five separately reported 200ms samples. */
        double elapsed;
        (void) run_put(put_src, put_out, side, 0.200, &elapsed);
        (void) run_prep(prep_src, prep_out, side, 0.200, &elapsed);
        for (int trial = 1; trial <= 5; trial++) {
            uint64_t calls = run_put(put_src, put_out, side, 0.200, &elapsed);
            double put_ns = elapsed * 1.0e9 / (double) calls;
            printf("put,%dx%d,%d,%" PRIu64 ",%.6f,%.3f\n", side, side, trial, calls, elapsed, put_ns);
            calls = run_prep(prep_src, prep_out, side, 0.200, &elapsed);
            double prep_ns = elapsed * 1.0e9 / (double) calls;
            printf("prep,%dx%d,%d,%" PRIu64 ",%.6f,%.3f\n", side, side, trial, calls, elapsed, prep_ns);
        }
        free(put_out);
        free(prep_out);
    }
    free((void *) convolve_ref);
    free((void *) compound_ref);
    fprintf(stderr, "bench sink=%" PRIu64 "\n", result_sink);
    return 0;
}

int main(int argc, char **argv) {
    const uint16_t endian_probe = 1;
    if (*(const uint8_t *) &endian_probe != 1) {
        fprintf(stderr, "fixture files are little-endian; this harness requires a little-endian host\n");
        return 2;
    }
    const char *dir = NULL;
    int mode = 0; /* 1=verify, 2=bench */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bench")) {
            if (mode) {
                fprintf(stderr, "choose exactly one of --verify or --bench\n");
                return 2;
            }
            mode = 2;
        } else if (!strcmp(argv[i], "--verify")) {
            if (mode) {
                fprintf(stderr, "choose exactly one of --verify or --bench\n");
                return 2;
            }
            mode = 1;
        } else if (!strcmp(argv[i], "--fixtures") && i + 1 < argc) {
            dir = argv[++i];
        } else {
            fprintf(stderr, "usage: %s (--verify|--bench) --fixtures DIR\n", argv[0]);
            return 2;
        }
    }
    if (!mode || !dir) {
        fprintf(stderr, "usage: %s (--verify|--bench) --fixtures DIR\n", argv[0]);
        return 2;
    }
    if (mode == 1) return verify(dir);
    if (verify(dir) != 0) return 1; /* Never time an unverified fixture set. */
    return benchmark(dir);
}
