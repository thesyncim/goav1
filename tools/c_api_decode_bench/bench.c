#define _POSIX_C_SOURCE 200809L

#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/md5_utils.h"

#if defined(GOAV1_DECODER_AOM)
#include "aom/aom_decoder.h"
#include "aom/aomdx.h"
#include "common/rawenc.h"
#elif defined(GOAV1_DECODER_DAV1D)
#include <dav1d/dav1d.h>
#else
#error "Build with GOAV1_DECODER_AOM or GOAV1_DECODER_DAV1D"
#endif

#ifndef GOAV1_BUILD_FLAGS
#define GOAV1_BUILD_FLAGS "unknown"
#endif
#ifndef GOAV1_AOM_SOURCE_COMMIT
#define GOAV1_AOM_SOURCE_COMMIT "unknown"
#endif
#ifndef GOAV1_DAV1D_REFERENCE_COMMIT
#define GOAV1_DAV1D_REFERENCE_COMMIT "unknown"
#endif

enum {
  EXPECTED_CLIPS = 18,
  EXPECTED_VISIBLE_FRAMES = 864,
  WARMUPS = 1,
  SAMPLES = 9,
  MAX_FLUSH_ROUNDS = 16
};

typedef struct {
  char *name;
  char *path;
  uint8_t *bytes;
  size_t size;
  char sha256[65];
} Clip;

typedef struct {
  Clip *clips;
  size_t count;
} Corpus;

typedef struct {
  const uint8_t *bytes;
  size_t size;
  size_t position;
  size_t packet_count;
  uint16_t width;
  uint16_t height;
  uint32_t declared_packets;
} IVFReader;

typedef struct {
  uint64_t observed;
  size_t packets;
  uint32_t declared_packets;
  uint16_t width;
  uint16_t height;
  int frames;
  uint8_t md5[16];
} DecodeResult;

typedef struct {
  int validate_only;
  uint32_t profile_repeat;
  const char *corpus_dir;
  const char *clip_name;
} Options;

static void set_error(char *dst, size_t cap, const char *message) {
  if (cap == 0) return;
  snprintf(dst, cap, "%s", message);
}

static uint16_t read_le16(const uint8_t *p) {
  return (uint16_t)((uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8));
}

static uint32_t read_le32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static char *copy_string(const char *src) {
  size_t n = strlen(src);
  char *dst = (char *)malloc(n + 1);
  if (dst != NULL) memcpy(dst, src, n + 1);
  return dst;
}

static char *join_path(const char *directory, const char *name) {
  size_t dir_len = strlen(directory);
  size_t name_len = strlen(name);
  size_t separator = dir_len != 0 && directory[dir_len - 1] != '/' ? 1 : 0;
  if (name_len > SIZE_MAX - separator - 1 ||
      dir_len > SIZE_MAX - separator - 1 - name_len) {
    return NULL;
  }
  size_t n = dir_len + name_len + separator + 1;
  char *path = (char *)malloc(n);
  if (path == NULL) return NULL;
  snprintf(path, n, "%s%s%s", directory, separator != 0 ? "/" : "", name);
  return path;
}

typedef struct {
  uint32_t state[8];
  uint64_t bits;
  uint8_t block[64];
  size_t used;
} SHA256Context;

static uint32_t rotate_right(uint32_t value, unsigned count) {
  return (value >> count) | (value << (32U - count));
}

static void sha256_transform(SHA256Context *ctx, const uint8_t block[64]) {
  static const uint32_t k[64] = {
      UINT32_C(0x428a2f98), UINT32_C(0x71374491), UINT32_C(0xb5c0fbcf),
      UINT32_C(0xe9b5dba5), UINT32_C(0x3956c25b), UINT32_C(0x59f111f1),
      UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5), UINT32_C(0xd807aa98),
      UINT32_C(0x12835b01), UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
      UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe), UINT32_C(0x9bdc06a7),
      UINT32_C(0xc19bf174), UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786),
      UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc), UINT32_C(0x2de92c6f),
      UINT32_C(0x4a7484aa), UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
      UINT32_C(0x983e5152), UINT32_C(0xa831c66d), UINT32_C(0xb00327c8),
      UINT32_C(0xbf597fc7), UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147),
      UINT32_C(0x06ca6351), UINT32_C(0x14292967), UINT32_C(0x27b70a85),
      UINT32_C(0x2e1b2138), UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
      UINT32_C(0x650a7354), UINT32_C(0x766a0abb), UINT32_C(0x81c2c92e),
      UINT32_C(0x92722c85), UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b),
      UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3), UINT32_C(0xd192e819),
      UINT32_C(0xd6990624), UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
      UINT32_C(0x19a4c116), UINT32_C(0x1e376c08), UINT32_C(0x2748774c),
      UINT32_C(0x34b0bcb5), UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a),
      UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3), UINT32_C(0x748f82ee),
      UINT32_C(0x78a5636f), UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
      UINT32_C(0x90befffa), UINT32_C(0xa4506ceb), UINT32_C(0xbef9a3f7),
      UINT32_C(0xc67178f2)};
  uint32_t words[64];
  for (size_t i = 0; i < 16; ++i) {
    size_t j = i * 4;
    words[i] = ((uint32_t)block[j] << 24) |
               ((uint32_t)block[j + 1] << 16) |
               ((uint32_t)block[j + 2] << 8) | (uint32_t)block[j + 3];
  }
  for (size_t i = 16; i < 64; ++i) {
    uint32_t x = words[i - 15];
    uint32_t y = words[i - 2];
    uint32_t s0 = rotate_right(x, 7) ^ rotate_right(x, 18) ^ (x >> 3);
    uint32_t s1 = rotate_right(y, 17) ^ rotate_right(y, 19) ^ (y >> 10);
    words[i] = words[i - 16] + s0 + words[i - 7] + s1;
  }

  uint32_t a = ctx->state[0];
  uint32_t b = ctx->state[1];
  uint32_t c = ctx->state[2];
  uint32_t d = ctx->state[3];
  uint32_t e = ctx->state[4];
  uint32_t f = ctx->state[5];
  uint32_t g = ctx->state[6];
  uint32_t h = ctx->state[7];
  for (size_t i = 0; i < 64; ++i) {
    uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
    uint32_t choice = (e & f) ^ ((~e) & g);
    uint32_t temp1 = h + s1 + choice + k[i] + words[i];
    uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
    uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    uint32_t temp2 = s0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }
  ctx->state[0] += a;
  ctx->state[1] += b;
  ctx->state[2] += c;
  ctx->state[3] += d;
  ctx->state[4] += e;
  ctx->state[5] += f;
  ctx->state[6] += g;
  ctx->state[7] += h;
}

static void sha256_init(SHA256Context *ctx) {
  static const uint32_t initial[8] = {
      UINT32_C(0x6a09e667), UINT32_C(0xbb67ae85), UINT32_C(0x3c6ef372),
      UINT32_C(0xa54ff53a), UINT32_C(0x510e527f), UINT32_C(0x9b05688c),
      UINT32_C(0x1f83d9ab), UINT32_C(0x5be0cd19)};
  memcpy(ctx->state, initial, sizeof(initial));
  ctx->bits = 0;
  ctx->used = 0;
}

static void sha256_update(SHA256Context *ctx, const uint8_t *data, size_t n) {
  while (n != 0) {
    size_t available = sizeof(ctx->block) - ctx->used;
    size_t chunk = n < available ? n : available;
    memcpy(ctx->block + ctx->used, data, chunk);
    ctx->used += chunk;
    data += chunk;
    n -= chunk;
    ctx->bits += (uint64_t)chunk * 8;
    if (ctx->used == sizeof(ctx->block)) {
      sha256_transform(ctx, ctx->block);
      ctx->used = 0;
    }
  }
}

static void sha256_final(SHA256Context *ctx, uint8_t digest[32]) {
  uint64_t bits = ctx->bits;
  ctx->block[ctx->used++] = 0x80;
  if (ctx->used > 56) {
    memset(ctx->block + ctx->used, 0, sizeof(ctx->block) - ctx->used);
    sha256_transform(ctx, ctx->block);
    ctx->used = 0;
  }
  memset(ctx->block + ctx->used, 0, 56 - ctx->used);
  for (size_t i = 0; i < 8; ++i) {
    ctx->block[63 - i] = (uint8_t)(bits >> (i * 8));
  }
  sha256_transform(ctx, ctx->block);
  for (size_t i = 0; i < 8; ++i) {
    digest[i * 4] = (uint8_t)(ctx->state[i] >> 24);
    digest[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
    digest[i * 4 + 2] = (uint8_t)(ctx->state[i] >> 8);
    digest[i * 4 + 3] = (uint8_t)ctx->state[i];
  }
}

static void bytes_to_hex(const uint8_t *bytes, size_t n, char *out) {
  static const char hex[] = "0123456789abcdef";
  for (size_t i = 0; i < n; ++i) {
    out[i * 2] = hex[bytes[i] >> 4];
    out[i * 2 + 1] = hex[bytes[i] & 0x0f];
  }
  out[n * 2] = '\0';
}

static int is_ivf_name(const char *name) {
  size_t n = strlen(name);
  return n > 4 && strcmp(name + n - 4, ".ivf") == 0;
}

static int compare_names(const void *a, const void *b) {
  const char *const *lhs = (const char *const *)a;
  const char *const *rhs = (const char *const *)b;
  return strcmp(*lhs, *rhs);
}

static int compare_clips(const void *a, const void *b) {
  const Clip *lhs = (const Clip *)a;
  const Clip *rhs = (const Clip *)b;
  return strcmp(lhs->name, rhs->name);
}

static int read_entire_file(const char *path, uint8_t **bytes, size_t *size,
                            char *error, size_t error_cap) {
  FILE *file = fopen(path, "rb");
  if (file == NULL) {
    snprintf(error, error_cap, "open %s: %s", path, strerror(errno));
    return -1;
  }
  if (fseek(file, 0, SEEK_END) != 0) {
    snprintf(error, error_cap, "seek %s: %s", path, strerror(errno));
    fclose(file);
    return -1;
  }
  long end = ftell(file);
  if (end < 0) {
    snprintf(error, error_cap, "tell %s: %s", path, strerror(errno));
    fclose(file);
    return -1;
  }
  if (fseek(file, 0, SEEK_SET) != 0) {
    snprintf(error, error_cap, "rewind %s: %s", path, strerror(errno));
    fclose(file);
    return -1;
  }
  if ((uintmax_t)end > (uintmax_t)SIZE_MAX) {
    snprintf(error, error_cap, "file too large: %s", path);
    fclose(file);
    return -1;
  }
  size_t n = (size_t)end;
  uint8_t *data = (uint8_t *)malloc(n == 0 ? 1 : n);
  if (data == NULL) {
    snprintf(error, error_cap, "allocate %zu bytes for %s", n, path);
    fclose(file);
    return -1;
  }
  if (n != 0 && fread(data, 1, n, file) != n) {
    snprintf(error, error_cap, "read %s: %s", path,
             ferror(file) ? strerror(errno) : "unexpected end of file");
    free(data);
    fclose(file);
    return -1;
  }
  if (fclose(file) != 0) {
    snprintf(error, error_cap, "close %s: %s", path, strerror(errno));
    free(data);
    return -1;
  }
  *bytes = data;
  *size = n;
  return 0;
}

static int load_corpus(const char *directory, const char *selected,
                       Corpus *corpus, char *error, size_t error_cap) {
  DIR *dir = opendir(directory);
  if (dir == NULL) {
    snprintf(error, error_cap, "open corpus directory %s: %s", directory,
             strerror(errno));
    return -1;
  }
  char **names = NULL;
  size_t name_count = 0;
  size_t name_capacity = 0;
  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    if (!is_ivf_name(entry->d_name)) continue;
    size_t n = strlen(entry->d_name);
    size_t stem_len = n - 4;
    if (selected != NULL && (strlen(selected) != stem_len ||
                             memcmp(selected, entry->d_name, stem_len) != 0)) {
      continue;
    }
    if (name_count == name_capacity) {
      size_t next = name_capacity == 0 ? 8 : name_capacity * 2;
      if (next < name_capacity || next > SIZE_MAX / sizeof(*names)) {
        set_error(error, error_cap, "too many corpus filenames");
        closedir(dir);
        goto fail_names;
      }
      char **grown = (char **)realloc(names, next * sizeof(*names));
      if (grown == NULL) {
        set_error(error, error_cap, "allocate corpus filename list");
        closedir(dir);
        goto fail_names;
      }
      names = grown;
      name_capacity = next;
    }
    names[name_count] = copy_string(entry->d_name);
    if (names[name_count] == NULL) {
      set_error(error, error_cap, "allocate corpus filename");
      closedir(dir);
      goto fail_names;
    }
    ++name_count;
  }
  closedir(dir);
  if (name_count == 0) {
    snprintf(error, error_cap, "no .ivf files found in %s", directory);
    goto fail_names;
  }
  qsort(names, name_count, sizeof(*names), compare_names);
  Clip *clips = (Clip *)calloc(name_count, sizeof(*clips));
  if (clips == NULL) {
    set_error(error, error_cap, "allocate clip list");
    goto fail_names;
  }
  for (size_t i = 0; i < name_count; ++i) {
    size_t n = strlen(names[i]);
    clips[i].name = (char *)malloc(n - 3);
    if (clips[i].name == NULL) {
      set_error(error, error_cap, "allocate clip name");
      goto fail_clips;
    }
    memcpy(clips[i].name, names[i], n - 4);
    clips[i].name[n - 4] = '\0';
    clips[i].path = join_path(directory, names[i]);
    if (clips[i].path == NULL) {
      set_error(error, error_cap, "allocate clip path");
      goto fail_clips;
    }
    if (read_entire_file(clips[i].path, &clips[i].bytes, &clips[i].size, error,
                         error_cap) != 0) {
      goto fail_clips;
    }
    SHA256Context sha;
    uint8_t digest[32];
    sha256_init(&sha);
    sha256_update(&sha, clips[i].bytes, clips[i].size);
    sha256_final(&sha, digest);
    bytes_to_hex(digest, sizeof(digest), clips[i].sha256);
    free(names[i]);
    names[i] = NULL;
  }
  free(names);
  corpus->clips = clips;
  corpus->count = name_count;
  qsort(corpus->clips, corpus->count, sizeof(*corpus->clips), compare_clips);
  return 0;

fail_clips:
  for (size_t i = 0; i < name_count; ++i) {
    free(clips[i].name);
    free(clips[i].path);
    free(clips[i].bytes);
  }
  free(clips);
fail_names:
  for (size_t i = 0; i < name_count; ++i) free(names[i]);
  free(names);
  return -1;
}

static void free_corpus(Corpus *corpus) {
  for (size_t i = 0; i < corpus->count; ++i) {
    free(corpus->clips[i].name);
    free(corpus->clips[i].path);
    free(corpus->clips[i].bytes);
  }
  free(corpus->clips);
  corpus->clips = NULL;
  corpus->count = 0;
}

static int ivf_reader_init(IVFReader *reader, const Clip *clip, char *error,
                           size_t error_cap) {
  memset(reader, 0, sizeof(*reader));
  if (clip->size < 32) {
    set_error(error, error_cap, "IVF header is shorter than 32 bytes");
    return -1;
  }
  if (memcmp(clip->bytes, "DKIF", 4) != 0) {
    set_error(error, error_cap, "invalid IVF signature");
    return -1;
  }
  if (read_le16(clip->bytes + 4) != 0) {
    set_error(error, error_cap, "unsupported IVF version");
    return -1;
  }
  uint16_t header_size = read_le16(clip->bytes + 6);
  if (header_size < 32 || (size_t)header_size > clip->size) {
    set_error(error, error_cap, "invalid IVF header length");
    return -1;
  }
  if (memcmp(clip->bytes + 8, "AV01", 4) != 0) {
    set_error(error, error_cap, "IVF fourcc is not AV01");
    return -1;
  }
  reader->width = read_le16(clip->bytes + 12);
  reader->height = read_le16(clip->bytes + 14);
  if (reader->width == 0 || reader->height == 0) {
    set_error(error, error_cap, "IVF dimensions are zero");
    return -1;
  }
  reader->declared_packets = read_le32(clip->bytes + 24);
  reader->bytes = clip->bytes;
  reader->size = clip->size;
  reader->position = (size_t)header_size;
  return 0;
}

static int ivf_reader_next(IVFReader *reader, const uint8_t **packet,
                           size_t *packet_size, char *error,
                           size_t error_cap) {
  if (reader->position == reader->size) {
    if (reader->declared_packets != 0 &&
        reader->packet_count != (size_t)reader->declared_packets) {
      snprintf(error, error_cap, "IVF declares %u packets but contains %zu",
               reader->declared_packets, reader->packet_count);
      return -1;
    }
    return 0;
  }
  if (reader->position > reader->size ||
      reader->size - reader->position < 12) {
    set_error(error, error_cap, "truncated IVF frame header");
    return -1;
  }
  uint32_t length = read_le32(reader->bytes + reader->position);
  reader->position += 12;
  if (length == 0) {
    set_error(error, error_cap, "IVF contains an empty frame packet");
    return -1;
  }
  if ((size_t)length > reader->size - reader->position) {
    set_error(error, error_cap, "IVF packet length exceeds remaining input");
    return -1;
  }
  *packet = reader->bytes + reader->position;
  *packet_size = (size_t)length;
  reader->position += (size_t)length;
  ++reader->packet_count;
  return 1;
}

static int add_frame_plane_observation(uint64_t *sum, const uint8_t *data,
                                      ptrdiff_t stride, size_t width,
                                      size_t height, size_t bytes_per_sample,
                                      char *error, size_t error_cap) {
  if (width == 0 || height == 0) return 0;
  if (data == NULL || stride <= 0 ||
      width > SIZE_MAX / bytes_per_sample) {
    set_error(error, error_cap, "invalid decoded plane buffer");
    return -1;
  }
  size_t row_bytes = width * bytes_per_sample;
  if ((uintmax_t)stride < (uintmax_t)row_bytes) {
    set_error(error, error_cap, "decoded plane stride is shorter than a row");
    return -1;
  }
  size_t row_offset = height - 1;
  if (row_offset != 0 &&
      (uintmax_t)stride > (uintmax_t)(SIZE_MAX / row_offset)) {
    set_error(error, error_cap, "decoded plane offset overflows size_t");
    return -1;
  }
  size_t last_row = row_offset * (size_t)stride;
  if (last_row > SIZE_MAX - (row_bytes - 1)) {
    set_error(error, error_cap, "decoded plane end offset overflows size_t");
    return -1;
  }
  size_t last = last_row + row_bytes - 1;
  *sum += (uint64_t)data[0] + (uint64_t)data[last];
  return 0;
}

#if defined(GOAV1_DECODER_DAV1D)
static int add_dav1d_plane_md5(MD5Context *md5, const uint8_t *data,
                               ptrdiff_t stride, size_t width, size_t height,
                               size_t bytes_per_sample, char *error,
                               size_t error_cap) {
  if (width == 0 || height == 0) return 0;
  if (data == NULL || stride <= 0 ||
      width > SIZE_MAX / bytes_per_sample) {
    set_error(error, error_cap, "invalid decoded plane for MD5");
    return -1;
  }
  size_t row_bytes = width * bytes_per_sample;
  if ((uintmax_t)stride < (uintmax_t)row_bytes || row_bytes > UINT_MAX) {
    set_error(error, error_cap, "invalid decoded plane row size for MD5");
    return -1;
  }
  if (height > 1 &&
      (uintmax_t)stride > (uintmax_t)(SIZE_MAX / (height - 1))) {
    set_error(error, error_cap, "decoded plane MD5 offset overflows size_t");
    return -1;
  }
  for (size_t y = 0; y < height; ++y) {
    MD5Update(md5, (const md5byte *)(data + y * (size_t)stride),
              (unsigned)row_bytes);
  }
  return 0;
}
#endif

static void digest_to_hex(const uint8_t digest[16], char out[33]) {
  static const char hex[] = "0123456789abcdef";
  for (size_t i = 0; i < 16; ++i) {
    out[i * 2] = hex[digest[i] >> 4];
    out[i * 2 + 1] = hex[digest[i] & 0x0f];
  }
  out[32] = '\0';
}

static int read_sidecar_md5(const Clip *clip, char out[33], char *error,
                            size_t error_cap) {
  size_t path_len = strlen(clip->path);
  if (path_len < 4 || path_len == SIZE_MAX) {
    set_error(error, error_cap, "invalid clip path for MD5 sidecar");
    return -1;
  }
  char *path = (char *)malloc(path_len + 1);
  if (path == NULL) {
    set_error(error, error_cap, "allocate MD5 sidecar path");
    return -1;
  }
  memcpy(path, clip->path, path_len - 4);
  memcpy(path + path_len - 4, ".md5", 5);
  FILE *file = fopen(path, "rb");
  if (file == NULL) {
    snprintf(error, error_cap, "open MD5 sidecar %s: %s", path,
             strerror(errno));
    free(path);
    return -1;
  }
  free(path);
  char line[256];
  if (fgets(line, sizeof(line), file) == NULL) {
    set_error(error, error_cap, "read MD5 sidecar");
    fclose(file);
    return -1;
  }
  if (fclose(file) != 0) {
    set_error(error, error_cap, "close MD5 sidecar");
    return -1;
  }
  size_t n = 0;
  while (n < 32 && line[n] != '\0' && line[n] != '\n' && line[n] != '\r') {
    char c = line[n];
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
          (c >= 'A' && c <= 'F'))) {
      set_error(error, error_cap, "MD5 sidecar does not start with 32 hex digits");
      return -1;
    }
    out[n] = c >= 'A' && c <= 'F' ? (char)(c - 'A' + 'a') : c;
    ++n;
  }
  if (n != 32) {
    set_error(error, error_cap, "MD5 sidecar does not contain 32 hex digits");
    return -1;
  }
  out[32] = '\0';
  return 0;
}

static int compare_md5_hex(const uint8_t digest[16], const char expected[33]) {
  char actual[33];
  digest_to_hex(digest, actual);
  return strcmp(actual, expected) == 0;
}

static uint64_t monotonic_ns(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    perror("clock_gettime(CLOCK_MONOTONIC)");
    exit(EXIT_FAILURE);
  }
  return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
         (uint64_t)value.tv_nsec;
}

static size_t ceil_shift(size_t value, unsigned shift) {
  size_t mask = (((size_t)1) << shift) - 1;
  return (value + mask) >> shift;
}

#if defined(GOAV1_DECODER_AOM)

static const char *backend_name(void) { return "libaom"; }
static const char *backend_version(void) { return aom_codec_version_str(); }
static const char *backend_reference_commit(void) {
  return GOAV1_AOM_SOURCE_COMMIT;
}
static int backend_source_checkout_commit_verified(void) { return 1; }

static int observe_aom_image(const aom_image_t *image, DecodeResult *result,
                             char *error, size_t error_cap) {
  if (image->d_w == 0 || image->d_h == 0) {
    set_error(error, error_cap, "libaom produced an empty image");
    return -1;
  }
  uint64_t value = ((uint64_t)image->d_w << 32) | (uint64_t)image->d_h;
  size_t bytes_per_sample =
      (image->fmt & AOM_IMG_FMT_HIGHBITDEPTH) != 0 ? 2 : 1;
  for (int plane = 0; plane < 3; ++plane) {
    unsigned x_shift = plane == 0 ? 0 : image->x_chroma_shift;
    unsigned y_shift = plane == 0 ? 0 : image->y_chroma_shift;
    size_t width = ceil_shift((size_t)image->d_w, x_shift);
    size_t height = ceil_shift((size_t)image->d_h, y_shift);
    if (plane != 0 && image->monochrome) continue;
    if (add_frame_plane_observation(
            &value, image->planes[plane], (ptrdiff_t)image->stride[plane],
            width, height, bytes_per_sample, error, error_cap) != 0) {
      return -1;
    }
  }
  result->observed += value;
  ++result->frames;
  return 0;
}

static int consume_aom_frames(aom_codec_ctx_t *codec, int hash_frames,
                              MD5Context *md5, DecodeResult *result,
                              int *produced, char *error, size_t error_cap) {
  aom_codec_iter_t iter = NULL;
  aom_image_t *image;
  *produced = 0;
  while ((image = aom_codec_get_frame(codec, &iter)) != NULL) {
    if (observe_aom_image(image, result, error, error_cap) != 0) return -1;
    if (hash_frames) {
      const int planes[3] = {AOM_PLANE_Y, AOM_PLANE_U, AOM_PLANE_V};
      raw_update_image_md5(image, planes, image->monochrome ? 1 : 3, md5);
    }
    ++*produced;
  }
  return 0;
}

static int decode_clip(const Clip *clip, int hash_frames, DecodeResult *result,
                       char *error, size_t error_cap) {
  IVFReader reader;
  if (ivf_reader_init(&reader, clip, error, error_cap) != 0) return -1;
  result->width = reader.width;
  result->height = reader.height;
  result->declared_packets = reader.declared_packets;
  aom_codec_ctx_t codec;
  memset(&codec, 0, sizeof(codec));
  aom_codec_dec_cfg_t config;
  memset(&config, 0, sizeof(config));
  config.threads = 1;
  config.allow_lowbitdepth = 1;
  aom_codec_err_t status =
      aom_codec_dec_init(&codec, aom_codec_av1_dx(), &config, 0);
  if (status != AOM_CODEC_OK) {
    snprintf(error, error_cap, "libaom decoder init: %s",
             aom_codec_err_to_string(status));
    return -1;
  }
  int initialized = 1;
  int failed = 0;
  MD5Context md5;
  if (hash_frames) MD5Init(&md5);
  if (aom_codec_control(&codec, AV1D_SET_ROW_MT, 0U) != AOM_CODEC_OK) {
    snprintf(error, error_cap, "libaom AV1D_SET_ROW_MT: %s",
             aom_codec_error(&codec));
    failed = 1;
    goto done;
  }

  for (;;) {
    const uint8_t *packet = NULL;
    size_t packet_size = 0;
    int next = ivf_reader_next(&reader, &packet, &packet_size, error,
                               error_cap);
    if (next < 0) {
      failed = 1;
      goto done;
    }
    if (next == 0) break;
    status = aom_codec_decode(&codec, packet, packet_size, NULL);
    if (status != AOM_CODEC_OK) {
      snprintf(error, error_cap, "libaom decode packet %zu: %s",
               reader.packet_count, aom_codec_error(&codec));
      failed = 1;
      goto done;
    }
    int produced = 0;
    if (consume_aom_frames(&codec, hash_frames, &md5, result, &produced,
                           error, error_cap) != 0) {
      failed = 1;
      goto done;
    }
  }

  for (int round = 0; round < MAX_FLUSH_ROUNDS; ++round) {
    status = aom_codec_decode(&codec, NULL, 0, NULL);
    if (status != AOM_CODEC_OK) {
      snprintf(error, error_cap, "libaom flush: %s", aom_codec_error(&codec));
      failed = 1;
      goto done;
    }
    int produced = 0;
    if (consume_aom_frames(&codec, hash_frames, &md5, result, &produced,
                           error, error_cap) != 0) {
      failed = 1;
      goto done;
    }
    if (produced == 0) break;
    if (round == MAX_FLUSH_ROUNDS - 1) {
      set_error(error, error_cap, "libaom flush did not reach an empty output");
      failed = 1;
      goto done;
    }
  }
  result->packets = reader.packet_count;

done:
  if (initialized) {
    status = aom_codec_destroy(&codec);
    if (status != AOM_CODEC_OK && !failed) {
      snprintf(error, error_cap, "libaom decoder destroy: %s",
               aom_codec_err_to_string(status));
      failed = 1;
    }
  }
  if (!failed && hash_frames) MD5Final(result->md5, &md5);
  if (!failed && result->frames == 0) {
    set_error(error, error_cap, "libaom produced no visible frames");
    failed = 1;
  }
  return failed ? -1 : 0;
}

#elif defined(GOAV1_DECODER_DAV1D)

static const char *backend_name(void) { return "dav1d"; }
static const char *backend_version(void) { return dav1d_version(); }
static const char *backend_reference_commit(void) {
  return GOAV1_DAV1D_REFERENCE_COMMIT;
}
static int backend_source_checkout_commit_verified(void) { return 0; }

static void retain_preloaded_packet(const uint8_t *buffer, void *cookie) {
  (void)buffer;
  (void)cookie;
}

static int dav1d_plane_dimensions(const Dav1dPicture *picture, int plane,
                                  size_t *width, size_t *height) {
  unsigned x_shift = 0;
  unsigned y_shift = 0;
  if (plane != 0) {
    switch (picture->p.layout) {
      case DAV1D_PIXEL_LAYOUT_I400:
        *width = 0;
        *height = 0;
        return 0;
      case DAV1D_PIXEL_LAYOUT_I420:
        x_shift = 1;
        y_shift = 1;
        break;
      case DAV1D_PIXEL_LAYOUT_I422:
        x_shift = 1;
        break;
      case DAV1D_PIXEL_LAYOUT_I444:
        break;
      default:
        return -1;
    }
  }
  *width = ceil_shift((size_t)picture->p.w, x_shift);
  *height = ceil_shift((size_t)picture->p.h, y_shift);
  return 0;
}

static int observe_dav1d_picture(const Dav1dPicture *picture,
                                DecodeResult *result, int hash_frames,
                                MD5Context *md5, char *error,
                                size_t error_cap) {
  if (picture->p.w <= 0 || picture->p.h <= 0 ||
      (picture->p.bpc != 8 && picture->p.bpc != 10)) {
    set_error(error, error_cap, "dav1d produced invalid picture properties");
    return -1;
  }
  if (picture->p.layout < DAV1D_PIXEL_LAYOUT_I400 ||
      picture->p.layout > DAV1D_PIXEL_LAYOUT_I444) {
    set_error(error, error_cap, "dav1d produced an unknown pixel layout");
    return -1;
  }
  size_t bytes_per_sample = picture->p.bpc > 8 ? 2 : 1;
  uint64_t value =
      ((uint64_t)(uint32_t)picture->p.w << 32) | (uint64_t)(uint32_t)picture->p.h;
  for (int plane = 0; plane < 3; ++plane) {
    size_t width = 0;
    size_t height = 0;
    if (dav1d_plane_dimensions(picture, plane, &width, &height) != 0) {
      set_error(error, error_cap, "dav1d picture has invalid subsampling");
      return -1;
    }
    if (width == 0 || height == 0) continue;
    const uint8_t *data = (const uint8_t *)picture->data[plane];
    ptrdiff_t stride = plane == 0 ? picture->stride[0] : picture->stride[1];
    if (add_frame_plane_observation(&value, data, stride, width, height,
                                    bytes_per_sample, error, error_cap) != 0) {
      return -1;
    }
    if (hash_frames &&
        add_dav1d_plane_md5(md5, data, stride, width, height, bytes_per_sample,
                            error, error_cap) != 0) {
      return -1;
    }
  }
  result->observed += value;
  ++result->frames;
  return 0;
}

static int drain_dav1d(Dav1dContext *decoder, int hash_frames, MD5Context *md5,
                       DecodeResult *result, int *produced, char *error,
                       size_t error_cap) {
  *produced = 0;
  for (;;) {
    Dav1dPicture picture;
    memset(&picture, 0, sizeof(picture));
    int status = dav1d_get_picture(decoder, &picture);
    if (status == DAV1D_ERR(EAGAIN)) return 0;
    if (status < 0) {
      snprintf(error, error_cap, "dav1d get_picture: %s", strerror(-status));
      return -1;
    }
    if (observe_dav1d_picture(&picture, result, hash_frames, md5, error,
                              error_cap) != 0) {
      dav1d_picture_unref(&picture);
      return -1;
    }
    dav1d_picture_unref(&picture);
    ++*produced;
  }
}

static int decode_clip(const Clip *clip, int hash_frames, DecodeResult *result,
                       char *error, size_t error_cap) {
  IVFReader reader;
  if (ivf_reader_init(&reader, clip, error, error_cap) != 0) return -1;
  result->width = reader.width;
  result->height = reader.height;
  result->declared_packets = reader.declared_packets;
  Dav1dSettings settings;
  dav1d_default_settings(&settings);
  settings.n_threads = 1;
  settings.max_frame_delay = 1;
  settings.output_invisible_frames = 0;
  settings.apply_grain = 1;
  Dav1dContext *decoder = NULL;
  int status = dav1d_open(&decoder, &settings);
  if (status < 0) {
    snprintf(error, error_cap, "dav1d open: %s", strerror(-status));
    return -1;
  }
  int failed = 0;
  MD5Context md5;
  if (hash_frames) MD5Init(&md5);
  for (;;) {
    const uint8_t *packet = NULL;
    size_t packet_size = 0;
    int next = ivf_reader_next(&reader, &packet, &packet_size, error,
                               error_cap);
    if (next < 0) {
      failed = 1;
      goto done;
    }
    if (next == 0) break;

    Dav1dData input;
    memset(&input, 0, sizeof(input));
    status = dav1d_data_wrap(&input, packet, packet_size,
                             retain_preloaded_packet, NULL);
    if (status < 0) {
      snprintf(error, error_cap, "dav1d data_wrap packet %zu: %s",
               reader.packet_count, strerror(-status));
      failed = 1;
      goto done;
    }
    while (input.sz != 0) {
      size_t size_before = input.sz;
      status = dav1d_send_data(decoder, &input);
      if (status == DAV1D_ERR(EAGAIN)) {
        int produced = 0;
        if (drain_dav1d(decoder, hash_frames, &md5, result, &produced, error,
                        error_cap) != 0) {
          failed = 1;
          break;
        }
        if (produced == 0) {
          set_error(error, error_cap,
                    "dav1d send_data/get_picture made no progress");
          failed = 1;
          break;
        }
        continue;
      }
      if (status < 0) {
        snprintf(error, error_cap, "dav1d send_data packet %zu: %s",
                 reader.packet_count, strerror(-status));
        failed = 1;
        break;
      }
      if (input.sz >= size_before) {
        set_error(error, error_cap, "dav1d accepted packet without consuming data");
        failed = 1;
        break;
      }
    }
    dav1d_data_unref(&input);
    if (failed) goto done;
    int produced = 0;
    if (drain_dav1d(decoder, hash_frames, &md5, result, &produced, error,
                    error_cap) != 0) {
      failed = 1;
      goto done;
    }
  }

  {
    int produced = 0;
    if (drain_dav1d(decoder, hash_frames, &md5, result, &produced, error,
                    error_cap) != 0) {
      failed = 1;
      goto done;
    }
  }
  result->packets = reader.packet_count;

done:
  dav1d_close(&decoder);
  if (!failed && hash_frames) MD5Final(result->md5, &md5);
  if (!failed && result->frames == 0) {
    set_error(error, error_cap, "dav1d produced no visible frames");
    failed = 1;
  }
  return failed ? -1 : 0;
}

#endif

typedef struct {
  DecodeResult preflight;
  char expected_md5[33];
  int sidecar_checked;
  uint64_t warmup_ns;
  uint64_t sample_ns[SAMPLES];
  uint64_t sample_observed[SAMPLES];
} ClipRun;

static void json_string(const char *value) {
  putchar('"');
  for (const unsigned char *p = (const unsigned char *)value; *p != '\0'; ++p) {
    switch (*p) {
      case '"':
        fputs("\\\"", stdout);
        break;
      case '\\':
        fputs("\\\\", stdout);
        break;
      case '\b':
        fputs("\\b", stdout);
        break;
      case '\f':
        fputs("\\f", stdout);
        break;
      case '\n':
        fputs("\\n", stdout);
        break;
      case '\r':
        fputs("\\r", stdout);
        break;
      case '\t':
        fputs("\\t", stdout);
        break;
      default:
        if (*p < 0x20) {
          printf("\\u%04x", (unsigned)*p);
        } else {
          putchar((int)*p);
        }
        break;
    }
  }
  putchar('"');
}

static void print_metadata(const Options *options, const Corpus *corpus) {
#if defined(__clang__)
  const char *compiler = "clang " __clang_version__;
#elif defined(__GNUC__)
  const char *compiler = "gcc " __VERSION__;
#else
  const char *compiler = __VERSION__;
#endif
  printf("{\"type\":\"metadata\",\"schema\":\"goav1-c-api-decode-v1\",");
  printf("\"backend\":");
  json_string(backend_name());
  printf(",\"backend_version\":");
  json_string(backend_version());
  printf(",\"backend_reference_commit\":");
  json_string(backend_reference_commit());
  printf(",\"backend_source_checkout_commit_verified\":%s",
         backend_source_checkout_commit_verified() ? "true" : "false");
  printf(",\"backend_binary_provenance_verified\":false");
  printf(",\"compiler\":");
  json_string(compiler);
  printf(",\"compiler_flags\":");
  json_string(GOAV1_BUILD_FLAGS);
  printf(",\"decoder_threads\":1,\"run_mode\":");
  json_string(options->profile_repeat != 0
                  ? "profile_repeat"
                  : options->validate_only ? "validate_only" : "sample");
  printf(",\"warmups\":%d,\"samples\":%d,\"profile_repetitions\":%u,",
         options->validate_only || options->profile_repeat != 0 ? 0 : WARMUPS,
         options->validate_only || options->profile_repeat != 0 ? 0 : SAMPLES,
         options->profile_repeat);
  printf("\"timings_collected\":%s,",
         !options->validate_only && options->profile_repeat == 0 ? "true"
                                                                 : "false");
#if defined(GOAV1_DECODER_AOM)
  printf("\"decode_cycle_scope\":");
  json_string(
      "IVF header and frame-record parsing; fresh libaom context open; decode "
      "all packets; drain visible frames; observe frame edges; destroy context");
#else
  printf("\"decode_cycle_scope\":");
  json_string(
      "IVF header and frame-record parsing; fresh dav1d context open; wrap "
      "each packet reference; decode all packets; drain visible frames; "
      "observe frame edges and unref pictures; close context");
#endif
  printf(",\"excluded_from_timing\":");
  json_string("input file reads, SHA-256, output MD5 preflight, sidecar reads, "
              "and sample result comparisons");
  printf(",\"input_storage\":");
  json_string("whole IVF files preloaded once; C packet payloads are slices "
              "into the immutable IVF bytes and are not copied");
  printf(",\"go_cold_boundary_note\":");
  json_string("goav1 NewDecoderFromIVF copies packet payloads; C APIs consume "
              "preloaded packet slices, so the cold boundaries differ");
  printf(",\"corpus_label\":\"exploratory_unmanifested\","
         "\"corpus_provenance\":\"not provided to this helper; do not infer "
         "source provenance\","
         "\"corpus_dir\":");
  json_string(options->corpus_dir);
  printf(",\"clips\":%zu,\"full_corpus_protocol\":%s,"
         "\"expected_full_corpus_clips\":%d,"
         "\"expected_full_corpus_visible_frames\":%d,",
         corpus->count, options->clip_name == NULL ? "true" : "false",
         EXPECTED_CLIPS, EXPECTED_VISIBLE_FRAMES);
#if defined(GOAV1_DECODER_AOM)
  printf("\"settings\":{\"threads\":1,\"row_mt\":false,"
         "\"allow_lowbitdepth\":true,\"film_grain\":\"libaom default\"},");
#else
  printf("\"settings\":{\"threads\":1,\"max_frame_delay\":1,"
         "\"output_invisible_frames\":false,\"apply_grain\":true,"
         "\"all_layers\":true},");
#endif
  printf("\"grain_scope\":");
  json_string("the unmanifested corpus has no declared grain coverage");
  printf(",\"md5_preflight\":\"every clip's visible output is compared to "
         "its stream-MD5 sidecar before sampling\",");
  printf("\"frame_observer\":");
  json_string("visible width and height plus first and last visible byte of "
              "each nonempty Y/U/V plane; accumulated per clip");
  printf("}\n");
}

static void print_clip_result(const Clip *clip, const ClipRun *run,
                              const Options *options) {
  char md5_hex[33];
  digest_to_hex(run->preflight.md5, md5_hex);
  printf("{\"type\":\"clip\",\"name\":");
  json_string(clip->name);
  printf(",\"ivf_bytes\":%zu,\"ivf_sha256\":\"%s\","
         "\"width\":%u,\"height\":%u,\"packet_count\":%zu,"
         "\"declared_packet_count\":%u,\"visible_frames\":%d,"
         "\"preflight_observed\":\"%" PRIu64 "\","
         "\"preflight_md5\":\"%s\",\"sidecar_md5\":\"%s\","
         "\"sidecar_check\":",
         clip->size, clip->sha256, (unsigned)run->preflight.width,
         (unsigned)run->preflight.height, run->preflight.packets,
         run->preflight.declared_packets, run->preflight.frames,
         run->preflight.observed, md5_hex, run->expected_md5);
  json_string(run->sidecar_checked ? "match" : "not_checked");
  printf(",\"validation_only\":%s",
         options->validate_only ? "true" : "false");
  if (options->profile_repeat != 0) {
    printf(",\"profile_repetitions\":%u,\"profile_visible_frames\":%" PRIu64
           ",\"profile_observed_per_decode\":\"%" PRIu64 "\"}\n",
           options->profile_repeat,
           (uint64_t)run->preflight.frames * options->profile_repeat,
           run->preflight.observed);
    return;
  }
  if (options->validate_only) {
    printf(",\"warmup_ns\":null,\"samples_ns\":[],"
           "\"sample_observed\":[],\"median_ns\":null}\n");
    return;
  }
  printf(",\"warmup_ns\":%" PRIu64 ",\"samples_ns\":[",
         run->warmup_ns);
  for (int i = 0; i < SAMPLES; ++i) {
    if (i != 0) putchar(',');
    printf("%" PRIu64, run->sample_ns[i]);
  }
  printf("],\"sample_observed\":[");
  for (int i = 0; i < SAMPLES; ++i) {
    if (i != 0) putchar(',');
    printf("\"%" PRIu64 "\"", run->sample_observed[i]);
  }
  uint64_t sorted[SAMPLES];
  memcpy(sorted, run->sample_ns, sizeof(sorted));
  for (int i = 1; i < SAMPLES; ++i) {
    uint64_t value = sorted[i];
    int j = i;
    while (j > 0 && sorted[j - 1] > value) {
      sorted[j] = sorted[j - 1];
      --j;
    }
    sorted[j] = value;
  }
  printf("],\"median_ns\":%" PRIu64 "}\n", sorted[SAMPLES / 2]);
}

static int same_decode_result(const DecodeResult *want,
                              const DecodeResult *got) {
  return want->frames == got->frames && want->packets == got->packets &&
         want->observed == got->observed && want->width == got->width &&
         want->height == got->height &&
         want->declared_packets == got->declared_packets;
}

static void usage(FILE *stream, const char *argv0) {
  fprintf(stream,
          "Usage: %s [--corpus DIR] [--clip NAME] [--validate-only]\n"
          "       %s [--corpus DIR] --clip NAME --profile-repeat COUNT\n"
          "The default protocol is one warmup and nine fresh-context samples "
          "per clip.\n"
          "Profile-repeat runs COUNT fresh-context decodes of one clip without "
          "per-iteration output.\n"
          "Set GOAV1_BENCH_CORPUS_DIR to override the default corpus path.\n",
          argv0, argv0);
}

static int parse_positive_u32(const char *text, uint32_t *value) {
  if (text == NULL || text[0] == '\0' || text[0] == '-') return -1;
  errno = 0;
  char *end = NULL;
  uintmax_t parsed = strtoumax(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
      parsed > UINT32_MAX) {
    return -1;
  }
  *value = (uint32_t)parsed;
  return 0;
}

static int parse_options(int argc, char **argv, Options *options) {
  options->validate_only = 0;
  options->profile_repeat = 0;
  options->corpus_dir = getenv("GOAV1_BENCH_CORPUS_DIR");
  if (options->corpus_dir == NULL || options->corpus_dir[0] == '\0') {
    options->corpus_dir = "testdata/benchcorpus";
  }
  options->clip_name = NULL;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      usage(stdout, argv[0]);
      return 1;
    }
    if (strcmp(argv[i], "--validate-only") == 0) {
      options->validate_only = 1;
      continue;
    }
    if (strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) {
      options->corpus_dir = argv[++i];
      continue;
    }
    if (strcmp(argv[i], "--clip") == 0 && i + 1 < argc) {
      options->clip_name = argv[++i];
      continue;
    }
    if (strcmp(argv[i], "--profile-repeat") == 0 && i + 1 < argc) {
      if (parse_positive_u32(argv[++i], &options->profile_repeat) != 0) {
        fprintf(stderr, "--profile-repeat requires an integer in 1..%u\n",
                UINT32_MAX);
        return -1;
      }
      continue;
    }
    fprintf(stderr, "unknown or incomplete argument: %s\n", argv[i]);
    usage(stderr, argv[0]);
    return -1;
  }
  if (options->profile_repeat != 0 &&
      (options->clip_name == NULL || options->validate_only)) {
    fprintf(stderr,
            "--profile-repeat requires --clip NAME and cannot be combined "
            "with --validate-only\n");
    return -1;
  }
  return 0;
}

int main(int argc, char **argv) {
  Options options;
  int parse_status = parse_options(argc, argv, &options);
  if (parse_status > 0) return EXIT_SUCCESS;
  if (parse_status < 0) return EXIT_FAILURE;

  char error[512] = {0};
  Corpus corpus = {0};
  if (load_corpus(options.corpus_dir, options.clip_name, &corpus, error,
                  sizeof(error)) != 0) {
    fprintf(stderr, "%s\n", error);
    return EXIT_FAILURE;
  }
  int full_corpus = options.clip_name == NULL;
  if (full_corpus && corpus.count != EXPECTED_CLIPS) {
    fprintf(stderr, "%s contains %zu clips; expected %d exploratory corpus "
                    "clips\n",
            options.corpus_dir, corpus.count, EXPECTED_CLIPS);
    free_corpus(&corpus);
    return EXIT_FAILURE;
  }
  if (options.clip_name != NULL && corpus.count != 1) {
    fprintf(stderr, "clip %s was not found exactly once in %s\n",
            options.clip_name, options.corpus_dir);
    free_corpus(&corpus);
    return EXIT_FAILURE;
  }

  ClipRun *runs = (ClipRun *)calloc(corpus.count, sizeof(*runs));
  if (runs == NULL) {
    fprintf(stderr, "allocate result rows\n");
    free_corpus(&corpus);
    return EXIT_FAILURE;
  }

  int total_frames = 0;
  int failed = 0;
  for (size_t i = 0; i < corpus.count; ++i) {
    Clip *clip = &corpus.clips[i];
    ClipRun *run = &runs[i];
    if (decode_clip(clip, 1, &run->preflight, error, sizeof(error)) != 0) {
      fprintf(stderr, "%s preflight decode: %s\n", clip->name, error);
      failed = 1;
      break;
    }
    if (read_sidecar_md5(clip, run->expected_md5, error, sizeof(error)) != 0) {
      fprintf(stderr, "%s sidecar: %s\n", clip->name, error);
      failed = 1;
      break;
    }
    if (!compare_md5_hex(run->preflight.md5, run->expected_md5)) {
      char got[33];
      digest_to_hex(run->preflight.md5, got);
      fprintf(stderr, "%s MD5 mismatch: got %s, sidecar %s\n", clip->name,
              got, run->expected_md5);
      failed = 1;
      break;
    }
    run->sidecar_checked = 1;
    total_frames += run->preflight.frames;
  }
  if (!failed && full_corpus && total_frames != EXPECTED_VISIBLE_FRAMES) {
    fprintf(stderr, "full corpus produced %d visible frames; expected %d\n",
            total_frames, EXPECTED_VISIBLE_FRAMES);
    failed = 1;
  }
  if (!failed && options.profile_repeat != 0) {
    Clip *clip = &corpus.clips[0];
    ClipRun *run = &runs[0];
    for (uint32_t repetition = 0; repetition < options.profile_repeat;
         ++repetition) {
      DecodeResult result = {0};
      error[0] = '\0';
      if (decode_clip(clip, 0, &result, error, sizeof(error)) != 0) {
        fprintf(stderr, "%s profile repetition %u decode: %s\n", clip->name,
                repetition + 1, error);
        failed = 1;
        break;
      }
      if (!same_decode_result(&run->preflight, &result)) {
        fprintf(stderr,
                "%s profile repetition %u output differs from preflight\n",
                clip->name, repetition + 1);
        failed = 1;
        break;
      }
    }
  } else if (!failed && !options.validate_only) {
    for (size_t i = 0; i < corpus.count; ++i) {
      Clip *clip = &corpus.clips[i];
      ClipRun *run = &runs[i];
      error[0] = '\0';
      DecodeResult warmup = {0};
      uint64_t start = monotonic_ns();
      int decode_status =
          decode_clip(clip, 0, &warmup, error, sizeof(error));
      run->warmup_ns = monotonic_ns() - start;
      if (decode_status != 0) {
        fprintf(stderr, "%s warmup decode: %s\n", clip->name, error);
        failed = 1;
        break;
      }
      if (!same_decode_result(&run->preflight, &warmup)) {
        fprintf(stderr, "%s warmup output differs from preflight\n",
                clip->name);
        failed = 1;
        break;
      }
      for (int sample = 0; sample < SAMPLES; ++sample) {
        DecodeResult result = {0};
        start = monotonic_ns();
        decode_status = decode_clip(clip, 0, &result, error, sizeof(error));
        run->sample_ns[sample] = monotonic_ns() - start;
        if (decode_status != 0) {
          fprintf(stderr, "%s sample %d decode: %s\n", clip->name,
                  sample + 1, error);
          failed = 1;
          break;
        }
        if (!same_decode_result(&run->preflight, &result)) {
          fprintf(stderr, "%s sample %d output differs from preflight\n",
                  clip->name, sample + 1);
          failed = 1;
          break;
        }
        run->sample_observed[sample] = result.observed;
      }
      if (failed) break;
    }
  }

  if (!failed) {
    print_metadata(&options, &corpus);
    for (size_t i = 0; i < corpus.count; ++i) {
      print_clip_result(&corpus.clips[i], &runs[i], &options);
    }
  }
  free(runs);
  free_corpus(&corpus);
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
