# C API full-clip decode sampler

This opt-in sampler compares the public Go decoder's full-clip path with the
libaom and dav1d C APIs. It defaults to the 18-clip, 864-visible-frame corpus
under testdata/benchcorpus. That local corpus has no source manifest, so its
results are exploratory and do not establish source provenance.

Build both backends against the pinned libaom 3.14.0 source/build and dav1d
1.5.3 installed through pkg-config:

~~~sh
bash tools/c_api_decode_bench/build.sh
~~~

The build expects the AOM source clone to be at the revision pinned in
scripts/verify_upstreams.sh and the configured AOM archive at
/tmp/goav1-quality-aom-build. It verifies the source checkout revision and
reports the decoder's runtime version, but does not independently attest the
archive's build provenance. Override those paths with GOAV1_AOM_SOURCE and
GOAV1_AOM_BUILD_DIR. The build directory defaults to
$TMPDIR/goav1-c-api-decode-bench-$CODEX_AGENT_ID.

Run a full sample or a validation-only preflight:

~~~sh
scripts/run_c_api_decode_bench.sh aom --corpus testdata/benchcorpus
scripts/run_c_api_decode_bench.sh dav1d --corpus testdata/benchcorpus
scripts/run_c_api_decode_bench.sh aom --corpus testdata/benchcorpus --validate-only
scripts/run_c_api_decode_bench.sh dav1d --corpus testdata/benchcorpus \
  --clip p720_inter_q32 --profile-repeat 1000
~~~

The helper reads each IVF file into memory before sampling. Each timed C run
parses its IVF header and packet records, opens a fresh single-thread decoder,
feeds all packets, drains every visible picture, observes the first and last
visible byte of each Y/U/V plane, releases dav1d pictures, and closes the
decoder. File reads, SHA-256 input identity, stream-MD5 verification, sidecar
reads, and result comparisons remain outside the timed interval. The dav1d
wrapper references the preloaded packet bytes and does not copy their payloads.

The Go cold sampler constructs from preloaded IVF bytes too, but
NewDecoderFromIVF copies packet payloads. The C cold boundary therefore has
different packet-storage work; the JSON metadata reports this limitation. A
fresh C decoder in a warm process is not a cold process launch. The one warmup
per clip lets the process and libraries initialize before nine measured
samples.

## Matched API sample

One matched run covered the full corpus: 18 clips of 48 frames each, 864 visible
frames total. Go, libaom, and dav1d each matched every clip's MD5 sidecar before
timing. Each path used one thread, one warmup, and nine whole-clip samples; the
reported aggregate is the sum of per-clip medians. File reads and MD5 checks
were outside the timed interval.

| Decoder path | Aggregate median sum |
| --- | ---: |
| Go cold `NewDecoderFromIVF` | 1758.601 ms |
| Go cold `NewDecoder(payloads)` | 1754.702 ms |
| Go warm `Reset` and decode | 1709.010 ms |
| libaom 3.14.0 C API | 851.404 ms |
| dav1d 1.5.3 C API | 542.140 ms |

The cold Go payload path extracts and copies IVF payloads before timing, then
times decoder construction, all `DecodeNext` calls, and `Close`; this was 0.22%
below the cold-IVF total, where constructor parsing and payload copies are
timed. The C helper times a fresh API decoder and packet parsing over preloaded
input bytes while viewing packet payloads without copying. Its cold boundary is
therefore similar but not identical to either Go path. These Go/C ratios were
2.07×/3.24× for cold IVF and 2.06×/3.24× for cold payloads against libaom and
dav1d respectively (above 1 means Go took longer). The warm Go path has no
matching warm-reset C measurement. The comparison excludes CLI startup and
does not isolate language or SIMD speed; both C libraries use architecture-
specific optimized code. The corpus remains exploratory because it has no
source manifest or provenance.

The helper verifies every C decode's visible-frame stream MD5 against its
existing sidecar before it times any clip. Full-corpus mode also requires 18
clips and 864 visible frames. Use --clip NAME to validate or sample one clip.

The Go side of the comparison also provides a cold pre-parsed-payload sample
and benchmark, `cold_preparsed_payload_decoder`, which separates IVF parsing
and payload copies from the timed decoder path. See
`docs/go127-simd-performance.md` for the full comparison and limits.

Output is newline-delimited JSON. The first row records backend/library and
compiler versions, exact compiler flags, decoder settings, decode-cycle scope,
corpus label, and the input-storage caveat. Each following row includes input
SHA-256, packet and visible-frame counts, preflight/sidecar MD5 values, the
warmup duration, nine raw sample durations, a median, and the sample
observation totals. --validate-only runs the full MD5 and frame-count
preflight without warmups or timed samples. The profiling mode requires one
--clip and a positive --profile-repeat count. It runs that many fresh-context
decodes after sidecar preflight, without per-iteration timing or output, so a
native profiler can attach while the decoder remains busy. Its output reports
the repetition count and decoded visible-frame total.

The current dav1d adapter accepts 8-bit and 10-bit pictures; 12-bit pictures
are rejected. The current corpus covers 8-bit clips and one 10-bit clip, and
contains no monochrome clip. Monochrome MD5 synthesis and 12-bit dav1d output
are outside this harness's validated scope.
