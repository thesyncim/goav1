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

/* dav1d src/mc.h low-bit-depth ABI; strides are bytes. */
typedef void (*put_fn_t)(uint8_t *, ptrdiff_t, const uint8_t *, ptrdiff_t,
                         int, int, int, int);
typedef void (*prep_fn_t)(int16_t *, const uint8_t *, ptrdiff_t,
                          int, int, int, int);
extern void dav1d_put_8tap_regular_smooth_8bpc_neon_i8mm(
    uint8_t *, ptrdiff_t, const uint8_t *, ptrdiff_t, int, int, int, int);
extern void dav1d_prep_8tap_regular_smooth_8bpc_neon_i8mm(
    int16_t *, const uint8_t *, ptrdiff_t, int, int, int, int);

static put_fn_t put_kernel = dav1d_put_8tap_regular_smooth_8bpc_neon_i8mm;
static prep_fn_t prep_kernel = dav1d_prep_8tap_regular_smooth_8bpc_neon_i8mm;
static volatile uint64_t result_sink;
static const int sides[] = { 8, 16, 32, 64, 128 };
static const int ref_side = 144;
static const int pad = 8;
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

static void path_name(char *buf, size_t n, const char *prefix, int side, const char *suffix) {
    if (snprintf(buf, n, "%s_%dx%d%s", prefix, side, side, suffix) >= (int) n) {
        fprintf(stderr, "fixture name too long\n");
        exit(2);
    }
}

static uint16_t get_u16le(const uint8_t *p) {
    return (uint16_t) p[0] | (uint16_t) ((uint16_t) p[1] << 8);
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
    const size_t ref_bytes = (size_t) ref_side * ref_side;
    const uint8_t *ref = read_file(dir, "ref_144x144_u8.bin", ref_bytes);
    const uint8_t *src = ref + (size_t) pad * ref_side + pad;
    int failures = 0;

    for (size_t si = 0; si < sizeof(sides) / sizeof(sides[0]); si++) {
        const int side = sides[si];
        const size_t count = (size_t) side * side;
        char name[128];
        uint8_t *put_out = calloc(count, sizeof(*put_out));
        int16_t *prep_out = calloc(count, sizeof(*prep_out));
        if (!put_out || !prep_out) die("calloc", "kernel outputs");
        path_name(name, sizeof(name), "put_expected", side, ".bin");
        const uint8_t *put_expected = read_file(dir, name, count);
        path_name(name, sizeof(name), "prep_expected", side, "_u16le.bin");
        const uint8_t *prep_expected = read_file(dir, name, count * sizeof(uint16_t));

        put_kernel(put_out, side, src, ref_side, side, side, mx, my);
        prep_kernel(prep_out, src, ref_side, side, side, mx, my);

        size_t put_diffs = 0, prep_diffs = 0;
        size_t first_put = count, first_prep = count;
        for (size_t i = 0; i < count; i++) {
            if (put_out[i] != put_expected[i]) {
                if (first_put == count) first_put = i;
                put_diffs++;
            }
            /* dav1d 8bpc PREP_BIAS is zero. Go's uint16 CONV_BUF retains the
               horizontal 2048 and vertical 4096 round biases. */
            const uint16_t prep_offset = (uint16_t) ((int32_t) prep_out[i] + 6144);
            const uint16_t expected = get_u16le(prep_expected + i * 2);
            if (prep_offset != expected) {
                if (first_prep == count) first_prep = i;
                prep_diffs++;
            }
        }
        fprintf(stderr, "verify,%dx%d,put_diffs=%zu,prep_diffs=%zu", side, side, put_diffs, prep_diffs);
        if (first_put < count) fprintf(stderr, ",first_put=%zu:%u/%u", first_put,
            (unsigned) put_out[first_put], (unsigned) put_expected[first_put]);
        if (first_prep < count) fprintf(stderr, ",first_prep=%zu:%u/%u", first_prep,
            (unsigned) ((int32_t) prep_out[first_prep] + 6144),
            (unsigned) get_u16le(prep_expected + first_prep * 2));
        fputc('\n', stderr);
        if (put_diffs || prep_diffs) failures++;

        result_sink += put_out[count - 1] + (uint16_t) prep_out[count - 1];
        free((void *) put_expected);
        free((void *) prep_expected);
        free(put_out);
        free(prep_out);
    }
    free((void *) ref);
    fprintf(stderr, "verify summary: %s; sink=%" PRIu64 "\n", failures ? "FAIL" : "PASS", result_sink);
    return failures ? 1 : 0;
}

static uint64_t run_put(const uint8_t *src, uint8_t *out, int side, double seconds, double *elapsed) {
    const double start = now_seconds();
    const double stop = start + seconds;
    uint64_t calls = 0;
    do {
        for (int i = 0; i < 16; i++)
            put_kernel(out, side, src, ref_side, side, side, mx, my);
        calls += 16;
    } while (now_seconds() < stop);
    *elapsed = now_seconds() - start;
    result_sink += out[(size_t) side * side - 1];
    return calls;
}

static uint64_t run_prep(const uint8_t *src, int16_t *out, int side, double seconds, double *elapsed) {
    const double start = now_seconds();
    const double stop = start + seconds;
    uint64_t calls = 0;
    do {
        for (int i = 0; i < 16; i++)
            prep_kernel(out, src, ref_side, side, side, mx, my);
        calls += 16;
    } while (now_seconds() < stop);
    *elapsed = now_seconds() - start;
    result_sink += (uint16_t) out[(size_t) side * side - 1];
    return calls;
}

static int benchmark(const char *dir) {
    const size_t ref_bytes = (size_t) ref_side * ref_side;
    const uint8_t *ref = read_file(dir, "ref_144x144_u8.bin", ref_bytes);
    const uint8_t *src = ref + (size_t) pad * ref_side + pad;
    puts("op,shape,trial,calls,elapsed_s,ns_per_call");

    for (size_t si = 0; si < sizeof(sides) / sizeof(sides[0]); si++) {
        const int side = sides[si];
        const size_t count = (size_t) side * side;
        uint8_t *put_out = calloc(count, sizeof(*put_out));
        int16_t *prep_out = calloc(count, sizeof(*prep_out));
        if (!put_out || !prep_out) die("calloc", "kernel outputs");
        double elapsed;
        (void) run_put(src, put_out, side, 0.200, &elapsed);
        (void) run_prep(src, prep_out, side, 0.200, &elapsed);
        for (int trial = 1; trial <= 5; trial++) {
            uint64_t calls = run_put(src, put_out, side, 0.200, &elapsed);
            printf("put,%dx%d,%d,%" PRIu64 ",%.6f,%.3f\n", side, side, trial, calls,
                   elapsed, elapsed * 1.0e9 / (double) calls);
            calls = run_prep(src, prep_out, side, 0.200, &elapsed);
            printf("prep,%dx%d,%d,%" PRIu64 ",%.6f,%.3f\n", side, side, trial, calls,
                   elapsed, elapsed * 1.0e9 / (double) calls);
        }
        free(put_out);
        free(prep_out);
    }
    free((void *) ref);
    fprintf(stderr, "bench sink=%" PRIu64 "\n", result_sink);
    return 0;
}

int main(int argc, char **argv) {
    const char *dir = NULL;
    int mode = 0; /* 1=verify, 2=bench */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bench")) {
            if (mode) return 2;
            mode = 2;
        } else if (!strcmp(argv[i], "--verify")) {
            if (mode) return 2;
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
    if (verify(dir) != 0) return 1;
    return benchmark(dir);
}
