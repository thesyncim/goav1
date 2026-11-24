// dav1d_validation_oracle decodes IVF files with libdav1d in its library
// default mode (strict_std_compliance = 0, the same as `dav1d --strict 0`) and
// reports, per file, which temporal units dav1d rejected and a hash of every
// picture it returned. decode_validation_diff_test.go compares the report with
// goav1 on mutated streams.
//
// Usage: oracle < list-of-ivf-paths
//
// Output, one line per input path:
//   <path> tus=<n> errs=<tu>,... pics=<tu>:<hash>,... msg=<first error log>
// A picture's tu is the temporal unit whose dav1d_send_data (or the final
// flush, tu=n) returned it. The hash is FNV-1a 64 over the little-endian
// 32-bit width and height followed by the visible Y, U and V rows (two bytes
// per sample above 8 bits), the layout hashFrameVisible uses.
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dav1d/dav1d.h>

static uint64_t fnv(uint64_t h, const uint8_t *p, size_t n) {
  for (size_t i = 0; i < n; i++) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static uint64_t hash_picture(const Dav1dPicture *p) {
  uint64_t h = 14695981039346656037ULL;
  const int w = p->p.w, ht = p->p.h;
  uint8_t dims[8];
  for (int i = 0; i < 4; i++) {
    dims[i] = (uint8_t)((unsigned)w >> (8 * i));
    dims[4 + i] = (uint8_t)((unsigned)ht >> (8 * i));
  }
  h = fnv(h, dims, 8);
  const int bps = p->p.bpc > 8 ? 2 : 1;
  const int ss_hor = p->p.layout == DAV1D_PIXEL_LAYOUT_I420 ||
                     p->p.layout == DAV1D_PIXEL_LAYOUT_I422;
  const int ss_ver = p->p.layout == DAV1D_PIXEL_LAYOUT_I420;
  for (int pl = 0; pl < 3; pl++) {
    if (pl && p->p.layout == DAV1D_PIXEL_LAYOUT_I400) break;
    const int pw = pl ? (w + ss_hor) >> ss_hor : w;
    const int ph = pl ? (ht + ss_ver) >> ss_ver : ht;
    const ptrdiff_t stride = p->stride[pl ? 1 : 0];
    const uint8_t *row = p->data[pl];
    for (int y = 0; y < ph; y++) {
      h = fnv(h, row + y * stride, (size_t)pw * (size_t)bps);
    }
  }
  return h;
}

typedef struct {
  char *buf;
  size_t len, cap;
} Str;

static void str_addf(Str *s, const char *fmt, ...) __attribute__((format(printf, 2, 3)));

#include <stdarg.h>
static void str_addf(Str *s, const char *fmt, ...) {
  for (;;) {
    va_list ap;
    va_start(ap, fmt);
    const size_t room = s->cap - s->len;
    const int n = vsnprintf(s->buf + s->len, room, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n < room) {
      s->len += (size_t)n;
      return;
    }
    s->cap = s->cap * 2 + (size_t)n + 64;
    s->buf = realloc(s->buf, s->cap);
    if (!s->buf) abort();
  }
}

// last_log collects dav1d's log messages for the current temporal unit; the
// first failing unit's messages are reported so disagreements can be traced to the dav1d check.
static char last_log[256];

static void capture_log(void *cookie, const char *fmt, va_list ap) {
  (void)cookie;
  const size_t n = strlen(last_log);
  if (n + 2 >= sizeof(last_log)) return;
  if (n) last_log[n] = '|';
  vsnprintf(last_log + n + !!n, sizeof(last_log) - n - !!n, fmt, ap);
}

static char first_msg[256];

static void note_error(Str *errs, int tu, int *nerr, int *last_err_tu) {
  if (*last_err_tu == tu) return;
  if (!*nerr) {
    size_t n = 0;
    for (const char *q = last_log; *q && n + 1 < sizeof(first_msg); q++) {
      if (*q == '\n') continue;
      first_msg[n++] = *q == ' ' ? '_' : *q;
    }
    first_msg[n] = 0;
    if (!n) strcpy(first_msg, "-");
  }
  str_addf(errs, "%s%d", (*nerr)++ ? "," : "", tu);
  *last_err_tu = tu;
}

static int drain(Dav1dContext *c, int tu, Str *pics, Str *errs, int *nerr_tu, int *last_err_tu) {
  for (int guard = 0; guard < 1024; guard++) {
    Dav1dPicture p;
    memset(&p, 0, sizeof(p));
    const int r = dav1d_get_picture(c, &p);
    if (r == DAV1D_ERR(EAGAIN)) return 0;
    if (r < 0) {
      note_error(errs, tu, nerr_tu, last_err_tu);
      if (r != DAV1D_ERR(EINVAL)) return r;
      continue;
    }
    str_addf(pics, "%s%d:%016" PRIx64, pics->len ? "," : "", tu, hash_picture(&p));
    dav1d_picture_unref(&p);
  }
  return 0;
}

static void run_file(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    printf("%s open-error\n", path);
    return;
  }
  fseek(f, 0, SEEK_END);
  const long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *bytes = malloc(sz > 0 ? (size_t)sz : 1);
  if (!bytes || fread(bytes, 1, (size_t)sz, f) != (size_t)sz) {
    printf("%s read-error\n", path);
    fclose(f);
    free(bytes);
    return;
  }
  fclose(f);
  if (sz < 32 || memcmp(bytes, "DKIF", 4)) {
    printf("%s not-ivf\n", path);
    free(bytes);
    return;
  }

  Dav1dSettings s;
  dav1d_default_settings(&s);
  s.n_threads = 1;
  s.max_frame_delay = 1;
  s.logger.callback = capture_log;
  last_log[0] = 0;
  strcpy(first_msg, "-");
  Dav1dContext *c = NULL;
  if (dav1d_open(&c, &s) < 0) {
    printf("%s open-dav1d-error\n", path);
    free(bytes);
    return;
  }

  Str pics = {0}, errs = {0};
  pics.cap = errs.cap = 256;
  pics.buf = calloc(1, pics.cap);
  errs.buf = calloc(1, errs.cap);
  int nerr = 0, last_err_tu = -1;
  size_t pos = (size_t)(bytes[6] | bytes[7] << 8);
  int tu = 0;
  while (pos + 12 <= (size_t)sz) {
    const size_t fsz = (size_t)bytes[pos] | (size_t)bytes[pos + 1] << 8 |
                       (size_t)bytes[pos + 2] << 16 | (size_t)bytes[pos + 3] << 24;
    pos += 12;
    if (fsz > (size_t)sz - pos) break;
    if (fsz == 0) {
      pos += fsz;
      tu++;
      continue;
    }
    last_log[0] = 0;
    Dav1dData data;
    uint8_t *dst = dav1d_data_create(&data, fsz);
    if (!dst) abort();
    memcpy(dst, bytes + pos, fsz);
    pos += fsz;
    for (int guard = 0; data.sz > 0 && guard < 1024; guard++) {
      const int r = dav1d_send_data(c, &data);
      if (r < 0 && r != DAV1D_ERR(EAGAIN)) {
        note_error(&errs, tu, &nerr, &last_err_tu);
        dav1d_data_unref(&data);
        break;
      }
      drain(c, tu, &pics, &errs, &nerr, &last_err_tu);
    }
    if (data.sz > 0) dav1d_data_unref(&data);
    drain(c, tu, &pics, &errs, &nerr, &last_err_tu);
    tu++;
  }
  drain(c, tu, &pics, &errs, &nerr, &last_err_tu);
  dav1d_close(&c);
  printf("%s tus=%d errs=%s pics=%s msg=%s\n", path, tu, errs.len ? errs.buf : "-",
         pics.len ? pics.buf : "-", first_msg);
  fflush(stdout);
  free(pics.buf);
  free(errs.buf);
  free(bytes);
}

int main(void) {
  char line[4096];
  while (fgets(line, sizeof(line), stdin)) {
    size_t n = strlen(line);
    while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
    if (n) run_file(line);
  }
  return 0;
}
