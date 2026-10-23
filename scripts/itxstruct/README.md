Agent measurement scripts from the fused inverse-transform program
(perf/itx-struct-N). Paths point at a session scratchpad; adjust before use.

- gates.sh tag "race pkgs" [decoder]: the full concurrent gate set
  (internal default+simd, root encoder subset, race, linux/amd64 simd vet +
  amd64 build, compiler-reports, alloc; with "decoder": corpus MD5+Workers
  default+simd FULLCLIP, dryrun-profiles, testvectors-fast, dryrun-fast).
  TestReconWavefrontSpeedup is wall-clock and flakes under this load; rerun
  it alone before treating it as real.
- ab.sh ivf runs binA binB: min instructions/cycles of interleaved
  single-thread decodes (GOMAXPROCS=1 -workers 1, /usr/bin/time -l).
- abenc.sh bitrate runs binA binB: the same for encbench 720p.
- ident.sh base new: encoder IVF byte-identity at 0.5/1/2 Mbps and
  1 Mbps -threads 4.
- realgate.sh bin [workers]: MD5 of every realcorpus stream (incl. aq0)
  against its dav1d .md5.
