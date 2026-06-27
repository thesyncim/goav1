//go:build goav1_oracle

// Package profiles holds the libaom profile-conformance regression test in its
// OWN test binary, fully isolated from the internal/av1/testvector oracle
// suite. Because it compiles into a different package (and therefore a
// different test binary) than internal/av1/testvector, it CANNOT share process
// state with TestLibaomFastFrameWorkDryRun or any other oracle test, so a
// decode here can never starve a fast-suite subtest of a reference surface.
//
// Each clip is decoded through a brand-new, fully local decoder: its own frame
// pool, frame-work state, motion store, surface-reference map, and scratch.
// Nothing is package-level and nothing is reused between clips, so the test is
// self-isolating even within its own binary.
//
// The decode path mirrors the byte-exact framework dry-run harness in
// internal/av1/testvector (runLibaomFrameWorkDryRun): residual decode +
// reconstruction + the full supported post-filter chain, with the per-block
// loop request configured to read intra/inter modes, motion vectors, motion
// modes, inter-intra, and compound blend. This is the path proven byte-exact
// against libaom. The public stream-runner path used by cmd/aom-go-dec now
// also reconstructs and post-filters byte-exact for these clips; it is guarded
// directly through the exported API by the conformance package
// (conformance/publicpath_test.go).
//
// The base all-intra clips are 64x64 (kf-max-dist=1) 3-frame
// profile-conformance bitstreams; the inter clips are 64x64 8-frame bitstreams
// (1 keyframe + 7 inter, kf-max-dist=30) encoded from a moving synthetic source
// so that inter prediction, MV scaling, OBMC and compound on NON-4:2:0 chroma
// subsampling (4:4:4 and 4:2:2) are exercised, with odd-edge profile-1 inter
// streams at both 8-bit and 10-bit. Profile 1 is covered at both 8-bit and
// 10-bit 4:4:4, plus 8/10-bit 4:4:4 screen-content clips that require luma and
// chroma palette prediction, 8/10-bit 4:4:4 CDEF/restoration clips, 8/10-bit
// 4:4:4 film-grain clips, a 10-bit 4:4:4 inter multi-tile clip, a profile-0
// hidden S_FRAME altref stream, and 8/10-bit 4:4:4 super-res clips that run
// the caller-owned full postfilter output path, including inter streams that
// reference super-res output and a high-bit-depth super-res + loop-restoration
// clip. All are committed under
// internal/av1/testvector/testdata/profiles/. libaom's published AV1 test-data
// ships no 4:4:4 (profile 1), 4:2:2 (profile 2) or 12-bit (profile 2) vectors,
// so these clips guard those decode paths, including 10-bit 4:2:2 inter
// prediction through CDEF and loop restoration.
//
// Regen recipe (run from a libaom v3.14.0 build tree, e.g. /tmp/aom-build):
//
//	# --- all-intra (kf-max-dist=1, 3 frames) ---
//	# 4:4:4 8-bit:
//	aomenc --i444 --width=64 --height=64 --limit=3 --ivf --profile=1 \
//	  --cpu-used=1 --end-usage=q --cq-level=32 --kf-max-dist=1 \
//	  --lag-in-frames=0 -o profile1-444-8bit-64x64.ivf src444.yuv
//	# 4:4:4 10-bit:
//	aomenc --i444 --width=64 --height=64 --limit=3 --ivf --profile=1 \
//	  --bit-depth=10 --input-bit-depth=10 --cpu-used=1 --end-usage=q \
//	  --cq-level=32 --kf-max-dist=1 --lag-in-frames=0 \
//	  -o profile1-444-10bit-64x64.ivf src444_10.yuv
//	# 4:2:0 8-bit S_FRAME altref:
//	aomenc --passes=2 --i420 --width=64 --height=64 --limit=8 --ivf \
//	  --profile=0 --cpu-used=4 --end-usage=q --cq-level=32 \
//	  --kf-max-dist=999 --auto-alt-ref=1 --lag-in-frames=25 \
//	  --sframe-dist=4 --sframe-mode=2 --enable-cdef=0 \
//	  --enable-restoration=0 -o profile0-420-8bit-sframe-64x64.ivf \
//	  sframe420.yuv
//	# 4:4:4 8-bit palette:
//	ffmpeg -f lavfi -i nullsrc=size=64x64:rate=1:duration=3 -frames:v 3 \
//	  -vf "geq=lum='if(lt(mod(X+N*4,16),8),64,192)':cb='if(lt(mod(Y+N*4,16),8),64,192)':cr='if(lt(mod(X+Y+N*4,32),16),64,192)',format=yuv444p" \
//	  -f rawvideo palette444.yuv
//	aomenc --i444 --width=64 --height=64 --limit=3 --ivf --profile=1 \
//	  --cpu-used=4 --end-usage=q --cq-level=20 --kf-max-dist=1 \
//	  --lag-in-frames=0 --tune-content=screen --enable-palette=1 \
//	  --enable-cdef=0 --enable-restoration=0 \
//	  -o profile1-444-8bit-palette-64x64.ivf palette444.yuv
//	# 4:4:4 10-bit palette:
//	python3 - <<'PY'
//	from pathlib import Path
//	w = h = 64
//	with Path("palette444_10.yuv").open("wb") as f:
//	    for n in range(3):
//	        for plane in range(3):
//	            for y in range(h):
//	                row = bytearray()
//	                for x in range(w):
//	                    if plane == 0:
//	                        v = 160 if ((x + n*4)//8 + y//8) % 2 == 0 else 832
//	                    elif plane == 1:
//	                        v = 224 if ((y + n*4)//8) % 2 == 0 else 768
//	                    else:
//	                        v = 256 if ((x + y + n*4)//8) % 2 == 0 else 704
//	                    row += bytes((v & 0xff, v >> 8))
//	                f.write(row)
//	PY
//	aomenc --i444 --width=64 --height=64 --limit=3 --ivf --profile=1 \
//	  --bit-depth=10 --input-bit-depth=10 --cpu-used=4 --end-usage=q \
//	  --cq-level=20 --kf-max-dist=1 --lag-in-frames=0 \
//	  --tune-content=screen --enable-palette=1 \
//	  --enable-cdef=0 --enable-restoration=0 \
//	  -o profile1-444-10bit-palette-64x64.ivf palette444_10.yuv
//	# 4:4:4 8-bit super-res:
//	aomenc --i444 --width=160 --height=128 --limit=4 --ivf --profile=1 \
//	  --cpu-used=4 --end-usage=q --cq-level=32 --kf-max-dist=1 \
//	  --lag-in-frames=0 --superres-mode=1 --superres-denominator=12 \
//	  --superres-kf-denominator=12 --enable-cdef=0 --enable-restoration=0 \
//	  -o profile1-444-8bit-superres-160x128.ivf src444_superres.yuv
//	# 4:4:4 8-bit super-res inter:
//	aomenc --i444 --width=160 --height=128 --limit=8 --ivf --profile=1 \
//	  --cpu-used=4 --end-usage=q --cq-level=32 --kf-max-dist=999 \
//	  --lag-in-frames=0 --superres-mode=1 --superres-denominator=12 \
//	  --superres-kf-denominator=12 --enable-cdef=0 --enable-restoration=0 \
//	  -o profile1-444-8bit-superres-inter-160x128.ivf src444_superres.yuv
//	# 4:4:4 10-bit super-res:
//	aomenc --i444 --width=160 --height=128 --limit=4 --ivf --profile=1 \
//	  --bit-depth=10 --input-bit-depth=10 --cpu-used=4 --end-usage=q \
//	  --cq-level=32 --kf-max-dist=1 --lag-in-frames=0 --superres-mode=1 \
//	  --superres-denominator=12 --superres-kf-denominator=12 \
//	  --enable-cdef=0 --enable-restoration=0 \
//	  -o profile1-444-10bit-superres-160x128.ivf src444_10_superres.yuv
//	# 4:4:4 10-bit super-res + loop restoration:
//	python3 - <<'PY'
//	from pathlib import Path
//	w, h, frames = 160, 128, 4
//	with Path("src444_10_superres_restoration.yuv").open("wb") as f:
//	    for n in range(frames):
//	        for plane in range(3):
//	            for y in range(h):
//	                row = bytearray()
//	                for x in range(w):
//	                    if plane == 0:
//	                        v = 96 + ((x*9 + y*7 + n*37 + ((x//8) ^ (y//8))*29) % 832)
//	                    elif plane == 1:
//	                        v = 128 + ((x*5 + y*13 + n*41 + ((x+y)//11)*31) % 768)
//	                    else:
//	                        v = 160 + ((x*17 + y*3 + n*43 + ((x*2+y)//9)*23) % 704)
//	                    row += bytes((v & 0xff, v >> 8))
//	                f.write(row)
//	PY
//	aomenc --i444 --width=160 --height=128 --limit=4 --ivf --profile=1 \
//	  --bit-depth=10 --input-bit-depth=10 --cpu-used=4 --end-usage=q \
//	  --cq-level=30 --kf-max-dist=1 --lag-in-frames=0 --superres-mode=1 \
//	  --superres-denominator=12 --superres-kf-denominator=12 \
//	  --enable-cdef=1 --enable-restoration=1 \
//	  -o profile1-444-10bit-superres-restoration-160x128.ivf \
//	  src444_10_superres_restoration.yuv
//	# 4:4:4 10-bit super-res static inter:
//	python3 - <<'PY'
//	from pathlib import Path
//	w, h, frames = 160, 128, 8
//	with Path("src444_10_superres_static_inter.yuv").open("wb") as f:
//	    for n in range(frames):
//	        for plane in range(3):
//	            for y in range(h):
//	                row = bytearray()
//	                for x in range(w):
//	                    if plane == 0:
//	                        v = 96 + ((x*7 + y*5 + (x ^ y)*3) % 832)
//	                    elif plane == 1:
//	                        v = 160 + ((x*11 + y*3) % 704)
//	                    else:
//	                        v = 128 + ((x*5 + y*13) % 736)
//	                    row += bytes((v & 0xff, v >> 8))
//	                f.write(row)
//	PY
//	aomenc --i444 --width=160 --height=128 --limit=8 --ivf --profile=1 \
//	  --bit-depth=10 --input-bit-depth=10 --cpu-used=4 --end-usage=q \
//	  --cq-level=32 --kf-max-dist=999 --lag-in-frames=0 \
//	  --superres-mode=1 --superres-denominator=12 \
//	  --superres-kf-denominator=12 --enable-cdef=0 --enable-restoration=0 \
//	  --loopfilter-control=0 --enable-obmc=0 --enable-global-motion=0 \
//	  --enable-warped-motion=0 --enable-dual-filter=0 --enable-masked-comp=0 \
//	  --enable-interintra-comp=0 --enable-onesided-comp=0 \
//	  --enable-interinter-wedge=0 --enable-diff-wtd-comp=0 \
//	  --enable-dist-wtd-comp=0 --enable-ref-frame-mvs=0 \
//	  --reduced-reference-set=1 --enable-tx64=0 --enable-rect-tx=0 \
//	  -o profile1-444-10bit-superres-inter-static-160x128.ivf \
//	  src444_10_superres_static_inter.yuv
//	# 4:4:4 8-bit CDEF + loop restoration:
//	aomenc --i444 --width=160 --height=128 --limit=4 --ivf --profile=1 \
//	  --cpu-used=2 --end-usage=q --cq-level=28 --kf-max-dist=1 \
//	  --lag-in-frames=0 --enable-cdef=1 --enable-restoration=1 \
//	  -o profile1-444-8bit-cdef-restoration-160x128.ivf src444_filters.yuv
//	# 4:4:4 8-bit odd edge-size CDEF + loop restoration:
//	ffmpeg -f lavfi -i nullsrc=size=130x130:rate=1:duration=3 -frames:v 3 \
//	  -vf "geq=lum='32+mod(X*7+Y*5+N*29+floor(X/9)*31+floor(Y/11)*17,208)':cb='40+mod(X*3+Y*11+N*23+floor((X+Y)/7)*19,184)':cr='48+mod(X*13+Y*2+N*31+floor((X*2+Y)/13)*23,176)',format=yuv444p" \
//	  -f rawvideo src444_8_edge_rest.yuv
//	aomenc --i444 --width=130 --height=130 --limit=3 --ivf --profile=1 \
//	  --cpu-used=4 --end-usage=q --cq-level=30 --kf-max-dist=1 \
//	  --lag-in-frames=0 --enable-cdef=1 --enable-restoration=1 \
//	  -o profile1-444-8bit-edge-cdef-restoration-130x130.ivf \
//	  src444_8_edge_rest.yuv
//	# 4:4:4 10-bit CDEF + loop restoration:
//	aomenc --i444 --width=160 --height=128 --limit=4 --ivf --profile=1 \
//	  --bit-depth=10 --input-bit-depth=10 --cpu-used=2 --end-usage=q \
//	  --cq-level=28 --kf-max-dist=1 --lag-in-frames=0 --enable-cdef=1 \
//	  --enable-restoration=1 \
//	  -o profile1-444-10bit-cdef-restoration-160x128.ivf src444_10_filters.yuv
//	# 4:4:4 10-bit odd edge-size CDEF + loop restoration:
//	aomenc --i444 --width=130 --height=130 --limit=3 --ivf --profile=1 \
//	  --bit-depth=10 --input-bit-depth=10 --cpu-used=4 --end-usage=q \
//	  --cq-level=30 --kf-max-dist=1 --lag-in-frames=0 --enable-cdef=1 \
//	  --enable-restoration=1 \
//	  -o profile1-444-10bit-edge-cdef-restoration-130x130.ivf \
//	  src444_10_edge_rest.yuv
//	# 4:4:4 8-bit film grain:
//	ffmpeg -f lavfi -i nullsrc=size=96x96:rate=1:duration=3 -frames:v 3 \
//	  -vf "geq=lum='48+mod(X*5+Y*3+N*19,160)':cb='64+mod(X*7+N*23,128)':cr='64+mod(Y*11+N*17,128)',format=yuv444p" \
//	  -f rawvideo filmgrain444.yuv
//	aomenc --i444 --width=96 --height=96 --limit=3 --ivf --profile=1 \
//	  --cpu-used=4 --end-usage=q --cq-level=32 --kf-max-dist=1 \
//	  --lag-in-frames=0 --enable-cdef=0 --enable-restoration=0 \
//	  --film-grain-test=1 \
//	  -o profile1-444-8bit-filmgrain-96x96.ivf filmgrain444.yuv
//	# 4:4:4 10-bit film grain:
//	python3 - <<'PY'
//	from pathlib import Path
//	w = h = 96
//	with Path("filmgrain444_10.yuv").open("wb") as f:
//	    for n in range(3):
//	        for plane in range(3):
//	            for y in range(h):
//	                row = bytearray()
//	                for x in range(w):
//	                    if plane == 0:
//	                        v = 192 + ((x*19 + y*11 + n*37) % 640)
//	                    elif plane == 1:
//	                        v = 256 + ((x*13 + y*5 + n*41) % 512)
//	                    else:
//	                        v = 256 + ((y*17 + x*3 + n*29) % 512)
//	                    row += bytes((v & 0xff, v >> 8))
//	                f.write(row)
//	PY
//	aomenc --i444 --width=96 --height=96 --limit=3 --ivf --profile=1 \
//	  --bit-depth=10 --input-bit-depth=10 --cpu-used=4 --end-usage=q \
//	  --cq-level=32 --kf-max-dist=1 --lag-in-frames=0 \
//	  --enable-cdef=0 --enable-restoration=0 --film-grain-test=1 \
//	  -o profile1-444-10bit-filmgrain-96x96.ivf filmgrain444_10.yuv
//	# 4:2:0 8-bit 128x128 superblock:
//	ffmpeg -f lavfi -i nullsrc=size=128x128:rate=1:duration=3 -frames:v 3 \
//	  -vf "geq=lum='64+mod(X*3+Y*2+N*17,128)':cb='64+mod(X*5+N*23,128)':cr='64+mod(Y*7+N*19,128)',format=yuv420p" \
//	  -f rawvideo sb128_420.yuv
//	aomenc --i420 --width=128 --height=128 --limit=3 --ivf --profile=0 \
//	  --cpu-used=4 --end-usage=q --cq-level=32 --kf-max-dist=1 \
//	  --lag-in-frames=0 --sb-size=128 --min-partition-size=128 \
//	  --max-partition-size=128 --enable-cdef=0 --enable-restoration=0 \
//	  -o superblock128-420-8bit-128x128.ivf sb128_420.yuv
//	# 4:2:2 8-bit:
//	aomenc --i422 ... --profile=2 ... -o profile2-422-8bit-64x64.ivf src422.yuv
//	# 4:2:2 10-bit odd edge-size:
//	aomenc --i422 --width=66 --height=66 --limit=3 --ivf --profile=2 \
//	  --bit-depth=10 --input-bit-depth=10 --cpu-used=4 --end-usage=q \
//	  --cq-level=32 --kf-max-dist=1 --lag-in-frames=0 \
//	  --enable-cdef=0 --enable-restoration=0 \
//	  -o profile2-422-10bit-66x66.ivf src422_10_edge.yuv
//	# 4:2:2 10-bit inter with active CDEF + loop restoration:
//	aomenc --i422 --width=160 --height=128 --limit=8 --ivf --profile=2 \
//	  --bit-depth=10 --input-bit-depth=10 --cpu-used=4 --end-usage=q \
//	  --cq-level=30 --kf-max-dist=999 --lag-in-frames=0 \
//	  --enable-cdef=1 --enable-restoration=1 \
//	  -o profile2-422-10bit-inter-cdef-restoration-160x128.ivf \
//	  src422_10_inter_filters.yuv
//	# 4:2:2 12-bit odd edge-size:
//	libaom encoder API, AOM_IMG_FMT_I42216, profile=2, bit_depth=12,
//	  input_bit_depth=12, chroma_subsampling_x=1, chroma_subsampling_y=0,
//	  cpu-used=4, q=34, kf-min/max-dist=1, lag-in-frames=0,
//	  enable-cdef=0, enable-restoration=0.
//	# 4:4:4 12-bit odd edge-size:
//	libaom encoder API, AOM_IMG_FMT_I44416, profile=2, bit_depth=12,
//	  input_bit_depth=12, cpu-used=8, q=34, kf-min/max-dist=1,
//	  lag-in-frames=0, enable-cdef=0, enable-restoration=0,
//	  enable-palette=0.
//	# 4:2:0 12-bit:
//	aomenc --i420 ... --profile=2 --bit-depth=12 --input-bit-depth=12 \
//	  --cq-level=40 ... -o profile2-420-12bit-64x64.ivf src420_12.yuv
//	# 4:2:0 12-bit odd edge-size with active CDEF:
//	aomenc --i420 --width=66 --height=66 --limit=3 --ivf --profile=2 \
//	  --bit-depth=12 --input-bit-depth=12 --cpu-used=4 --end-usage=q \
//	  --cq-level=36 --kf-max-dist=1 --lag-in-frames=0 \
//	  --enable-cdef=1 --enable-restoration=1 \
//	  -o profile2-420-12bit-66x66.ivf src420_12_edge.yuv
//	# 4:2:0 12-bit film grain:
//	aomenc --i420 --width=96 --height=96 --limit=3 --ivf --profile=2 \
//	  --bit-depth=12 --input-bit-depth=12 --cpu-used=4 --end-usage=q \
//	  --cq-level=40 --kf-max-dist=1 --lag-in-frames=0 \
//	  --enable-cdef=0 --enable-restoration=0 --film-grain-test=1 \
//	  -o profile2-420-12bit-filmgrain-96x96.ivf filmgrain420_12.yuv
//	# 4:2:0 12-bit super-res:
//	aomenc --i420 --width=160 --height=128 --limit=4 --ivf --profile=2 \
//	  --bit-depth=12 --input-bit-depth=12 --cpu-used=4 --end-usage=q \
//	  --cq-level=36 --kf-max-dist=1 --lag-in-frames=0 \
//	  --superres-mode=1 --superres-denominator=12 \
//	  --superres-kf-denominator=12 --enable-cdef=0 --enable-restoration=0 \
//	  -o profile2-420-12bit-superres-160x128.ivf src420_12_superres.yuv
//
//	# --- inter (kf-max-dist=30, 8 frames, moving synthetic source) ---
//	# 4:4:4 8-bit inter:
//	aomenc --i444 --width=64 --height=64 --limit=8 --ivf --profile=1 \
//	  --cpu-used=4 --end-usage=q --cq-level=32 --kf-max-dist=30 \
//	  --lag-in-frames=0 -o profile1-444-8bit-inter-64x64.ivf src444.yuv
//	# 4:4:4 8-bit odd edge-size inter:
//	ffmpeg -f lavfi -i nullsrc=size=130x130:rate=1:duration=5 -frames:v 5 \
//	  -vf "geq=lum='48+mod((X+N*7)*3+(Y+N*5)*2+floor((X+N*4)/13)*17,176)':cb='32+mod((X+N*9)*5+Y*3,192)':cr='40+mod(X*2+(Y+N*8)*7,184)',format=yuv444p" \
//	  -f rawvideo src444_8_edgemv.yuv
//	aomenc --i444 --width=130 --height=130 --limit=5 --ivf --profile=1 \
//	  --cpu-used=4 --end-usage=q --cq-level=32 --kf-max-dist=999 \
//	  --lag-in-frames=0 --enable-cdef=0 --enable-restoration=0 \
//	  -o profile1-444-8bit-edgemv-130x130.ivf src444_8_edgemv.yuv
//	# 4:4:4 8-bit multi-tile:
//	ffmpeg -f lavfi -i nullsrc=size=256x256:rate=1:duration=4 -frames:v 4 \
//	  -vf "geq=lum='32+mod(X*5+Y*3+N*31+floor(X/16)*23+floor(Y/16)*11,208)':cb='48+mod(X*7+Y*2+N*19+floor((X+Y)/12)*17,176)':cr='40+mod(X*3+Y*9+N*29+floor((X*2+Y)/15)*13,184)',format=yuv444p" \
//	  -f rawvideo src444_8_multitile.yuv
//	aomenc --i444 --width=256 --height=256 --limit=4 --ivf --profile=1 \
//	  --cpu-used=4 --end-usage=q --cq-level=32 --kf-max-dist=1 \
//	  --lag-in-frames=0 --tile-columns=1 --enable-cdef=0 \
//	  --enable-restoration=0 \
//	  -o profile1-444-8bit-multitile-2x1-256x256.ivf \
//	  src444_8_multitile.yuv
//	# 4:4:4 10-bit inter:
//	aomenc --i444 --width=64 --height=64 --limit=8 --ivf --profile=1 \
//	  --bit-depth=10 --input-bit-depth=10 --cpu-used=4 --end-usage=q \
//	  --cq-level=32 --kf-max-dist=30 --lag-in-frames=0 \
//	  -o profile1-444-10bit-inter-64x64.ivf src444_10.yuv
//	# 4:4:4 10-bit odd edge-size inter:
//	aomenc --i444 --width=130 --height=130 --limit=5 --ivf --profile=1 \
//	  --bit-depth=10 --input-bit-depth=10 --cpu-used=4 --end-usage=q \
//	  --cq-level=32 --kf-max-dist=999 --lag-in-frames=0 \
//	  --enable-cdef=0 --enable-restoration=0 \
//	  -o profile1-444-10bit-edgemv-130x130.ivf src444_10_edgemv.yuv
//	# 4:4:4 10-bit inter multi-tile:
//	python3 - <<'PY'
//	from pathlib import Path
//	w, h, frames = 128, 128, 5
//	with Path("src444_10_multitile_inter.yuv").open("wb") as f:
//	    for n in range(frames):
//	        sx, sy = n * 7, n * 5
//	        for plane in range(3):
//	            for y in range(h):
//	                row = bytearray()
//	                for x in range(w):
//	                    if plane == 0:
//	                        block = (((x + sx) // 8) ^ ((y + sy) // 8)) * 37
//	                        v = 96 + ((x*7 + y*5 + n*43 + block) % 832)
//	                    elif plane == 1:
//	                        block = (((x + n*9) // 7) + ((y + n*3) // 11)) * 29
//	                        v = 128 + ((x*13 + y*3 + n*31 + block) % 768)
//	                    else:
//	                        block = (((x + y + n*6) // 9) ^ ((x*2 + y) // 13)) * 31
//	                        v = 160 + ((x*5 + y*17 + n*47 + block) % 704)
//	                    row += bytes((v & 0xff, v >> 8))
//	                f.write(row)
//	PY
//	aomenc --i444 --width=128 --height=128 --limit=5 --ivf --profile=1 \
//	  --bit-depth=10 --input-bit-depth=10 --cpu-used=4 --end-usage=q \
//	  --cq-level=32 --kf-max-dist=999 --lag-in-frames=0 --tile-columns=1 \
//	  --enable-cdef=0 --enable-restoration=0 \
//	  -o profile1-444-10bit-multitile-inter-2x1-128x128.ivf \
//	  src444_10_multitile_inter.yuv
//	# 4:2:2 8-bit inter:
//	aomenc --i422 ... --profile=2 ... -o profile2-422-8bit-inter-64x64.ivf src422.yuv
//
//	# Golden per-frame MD5s come from `aomdec --rawvideo` (libaom
//	# test/md5_helper.h layout, which testvector.FrameMD5 reproduces): split the
//	# raw output into per-frame chunks (W*H*3 for 4:4:4, W*H + 2*(W/2*H) for
//	# 4:2:2), multiply by bytes/sample for high bit depth, and md5 each chunk.
package profiles

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/decoder"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/reconstruct"
	"github.com/thesyncim/goav1/internal/av1/testvector"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

type profileClip struct {
	name string
	file string

	frameMD5Hex []string

	wantSeqProfile        uint8
	wantBitDepth          uint8
	wantSubsamplingX      bool
	wantSubsamplingY      bool
	checkUse128x128SB     bool
	wantUse128x128SB      bool
	wantPaletteYBlocks    int
	wantPaletteUVBlocks   int
	wantCDEFFrames        int
	wantRestorationFrames int
	wantFilmGrainFrames   int
	wantInterFrames       int
	wantSwitchFrames      int
	wantTileCols          int
	wantTileRows          int
	wantTileGroupTiles    int

	// includeHiddenFrameMD5 makes frameMD5Hex cover every completed decoded
	// frame, including non-shown altref/S_FRAME outputs. Libaom's rawvideo
	// oracle emits those frames for this stream, while the older profile clips
	// keep comparing shown outputs only.
	includeHiddenFrameMD5 bool

	// superRes marks clips that signal frame super-resolution. Super-res
	// reallocates the displayed surface to a larger (upscaled) width than the
	// coded reconstruction, so it cannot run through the in-place publication
	// post-filter runner; these clips use the caller-owned ApplyCallerPostFilters
	// chain instead, exactly like a display/output consumer.
	superRes bool
}

const profileSuperResOutputGlobalBase = 256

var profileClips = []profileClip{
	{
		// Profile 0: 4:2:0 8-bit two-pass altref stream with a hidden S_FRAME.
		// The MD5 list uses libaom aomdec --rawvideo frame bytes mapped into
		// low-overhead decode-event order, including the hidden switch frame
		// and hidden altref frame.
		name: "profile0-420-8bit-sframe-64x64",
		file: "profile0-420-8bit-sframe-64x64.ivf",
		frameMD5Hex: []string{
			"a09b7a904d7faed040036f0dbbc52aa2",
			"63e25c36bd77d0908cb9eb2361fb39d1",
			"17c94cf420710324e592ffc3cd9ce556",
			"c5de5e0baeec848e384830398f528c61",
			"df4820bf2ae49c081156d90e9994c0c5",
			"c1896b50d88a6483656cb8ae38354ba1",
			"e8a806ae6385776252b7c74c8c7b6dd5",
			"dce197270249c6bb8bda4d7c86b9b6b8",
		},
		wantSeqProfile:        0,
		wantBitDepth:          8,
		wantSubsamplingX:      true,
		wantSubsamplingY:      true,
		wantInterFrames:       6,
		wantSwitchFrames:      1,
		includeHiddenFrameMD5: true,
	},
	{
		// Profile 1: 4:4:4 8-bit. SubsamplingX=false SubsamplingY=false.
		name: "profile1-444-8bit-64x64",
		file: "profile1-444-8bit-64x64.ivf",
		frameMD5Hex: []string{
			"00211cdc8f799c808849c955a318a0f5",
			"397ff01920ff514bc611ab49d76371c1",
			"f8fbfb25a42da47a7adb71510de9b178",
		},
		wantSeqProfile:   1,
		wantBitDepth:     8,
		wantSubsamplingX: false,
		wantSubsamplingY: false,
	},
	{
		// Profile 1: 4:4:4 8-bit screen-content clip with palette-coded blocks.
		name: "profile1-444-8bit-palette-64x64",
		file: "profile1-444-8bit-palette-64x64.ivf",
		frameMD5Hex: []string{
			"59e96ccf6f2faf403ae8f8b3214a202f",
			"9a9cfad7d6aead1fb110309d2edc0268",
			"6d7426c5bae2c5cca7c98b584e8057a6",
		},
		wantSeqProfile:      1,
		wantBitDepth:        8,
		wantSubsamplingX:    false,
		wantSubsamplingY:    false,
		wantPaletteYBlocks:  1,
		wantPaletteUVBlocks: 1,
	},
	{
		// Profile 1: 4:4:4 10-bit screen-content clip with palette-coded
		// blocks, guarding high-bit-depth luma and chroma palette prediction.
		name: "profile1-444-10bit-palette-64x64",
		file: "profile1-444-10bit-palette-64x64.ivf",
		frameMD5Hex: []string{
			"dcd2ab9753a24747f1b731d011e6478a",
			"b9aaba28bd6cc85fe157befe067cada9",
			"dc714a2fc421619ab9384c185cf21ab6",
		},
		wantSeqProfile:      1,
		wantBitDepth:        10,
		wantSubsamplingX:    false,
		wantSubsamplingY:    false,
		wantPaletteYBlocks:  1,
		wantPaletteUVBlocks: 1,
	},
	{
		// Profile 1: 4:4:4 10-bit. SubsamplingX=false SubsamplingY=false.
		name: "profile1-444-10bit-64x64",
		file: "profile1-444-10bit-64x64.ivf",
		frameMD5Hex: []string{
			"96322d4430f0d27243c17e2128cfd625",
			"571c0415f63a1f76e3796360aee3828a",
			"2f6e92f93a95cb4725b9c2f9484ed42e",
		},
		wantSeqProfile:   1,
		wantBitDepth:     10,
		wantSubsamplingX: false,
		wantSubsamplingY: false,
	},
	{
		// Profile 2: 4:2:2 8-bit. SubsamplingX=true SubsamplingY=false.
		name: "profile2-422-8bit-64x64",
		file: "profile2-422-8bit-64x64.ivf",
		frameMD5Hex: []string{
			"dabd492413632a810adeaf4e5d0c6d97",
			"eb6cf8d1d4d644686cf03c513acd5978",
			"84983e98afeef6692448e98fc5980431",
		},
		wantSeqProfile:   2,
		wantBitDepth:     8,
		wantSubsamplingX: true,
		wantSubsamplingY: false,
	},
	{
		// Profile 2: 4:2:2 10-bit odd 66x66 size, guarding high-bit-depth
		// chroma rows at non-4:2:0 frame edges.
		name: "profile2-422-10bit-66x66",
		file: "profile2-422-10bit-66x66.ivf",
		frameMD5Hex: []string{
			"8f6b2f543fdd300f6a09285bd35b2dea",
			"e6137fd48ed1c842cfef45b6a13d09e9",
			"d66f267df360523a90ae1f25a101f68c",
		},
		wantSeqProfile:   2,
		wantBitDepth:     10,
		wantSubsamplingX: true,
		wantSubsamplingY: false,
	},
	{
		// Profile 2: 4:2:2 12-bit odd 66x66 size, guarding the professional
		// profile's maximum bit depth on non-4:2:0 chroma rows.
		name: "profile2-422-12bit-66x66",
		file: "profile2-422-12bit-66x66.ivf",
		frameMD5Hex: []string{
			"f353805adbc4cd31d4ca47e06dc5ee37",
			"a5455437eb82ee1d56009edc35495d6c",
			"ede56f76fa8800b3e117f6f2829c433a",
		},
		wantSeqProfile:   2,
		wantBitDepth:     12,
		wantSubsamplingX: true,
		wantSubsamplingY: false,
	},
	{
		// Profile 2: 4:4:4 12-bit odd 66x66 size, guarding maximum bit depth
		// with no chroma subsampling.
		name: "profile2-444-12bit-66x66",
		file: "profile2-444-12bit-66x66.ivf",
		frameMD5Hex: []string{
			"949936ee55d10f23afd156c8aa5f9564",
			"e70adc26dabc35bd3cc294e2eeca4c5c",
			"b4951e75489506c04dfc110213b53031",
		},
		wantSeqProfile:   2,
		wantBitDepth:     12,
		wantSubsamplingX: false,
		wantSubsamplingY: false,
	},
	{
		// Profile 1: 4:4:4 8-bit INTER. 8 frames (1 keyframe + 7 inter,
		// kf-max-dist=30) from a moving synthetic source, so inter prediction,
		// MV scaling, OBMC and compound on non-4:2:0 chroma are exercised.
		name: "profile1-444-8bit-inter-64x64",
		file: "profile1-444-8bit-inter-64x64.ivf",
		frameMD5Hex: []string{
			"11d7f1cc66c0e0484ad4d7566d66df35",
			"0e590539ec2f2e33419f1de467ab9a18",
			"d829dfffec65fcd80746b20f1c22b21b",
			"2fdb71b5156533455f031e37a96b2a4d",
			"23f8e6ee901994b9cbd2cfb91e181eeb",
			"b69530f8be1ca111af9e9b4c29c3c984",
			"cc93559f8904df46c9a6f3223a3c1f30",
			"fafc19265b25a4380c50ae5a40e4eb27",
		},
		wantSeqProfile:   1,
		wantBitDepth:     8,
		wantSubsamplingX: false,
		wantSubsamplingY: false,
	},
	{
		// Profile 1: 4:4:4 8-bit odd 130x130 INTER, guarding byte-path edge
		// motion prediction and chroma padding on non-4:2:0 streams.
		name: "profile1-444-8bit-edgemv-130x130",
		file: "profile1-444-8bit-edgemv-130x130.ivf",
		frameMD5Hex: []string{
			"45b7de404cb0b5d469958768b8f5d479",
			"945ed5c3b0a1191eed1499b6f3425025",
			"35f885d3ec1b37fb61f896b78f4a7ec8",
			"e852ba422c8e6c6e0d4e9077c1751f8d",
			"956fbda68f31ead438b5835aa5f87980",
		},
		wantSeqProfile:   1,
		wantBitDepth:     8,
		wantSubsamplingX: false,
		wantSubsamplingY: false,
		wantInterFrames:  1,
	},
	{
		// Profile 1: 4:4:4 8-bit multi-tile, guarding non-4:2:0 tile layout,
		// split/reconstruction, and per-tile CDF retention.
		name: "profile1-444-8bit-multitile-2x1-256x256",
		file: "profile1-444-8bit-multitile-2x1-256x256.ivf",
		frameMD5Hex: []string{
			"afb8da499aa3e0394b61ae7859323bd7",
			"eaf99aece59fc4ecbfb556eee5fcd3fc",
			"7433db9f86a39411f616573b5735aad5",
			"3661ba977cd76893d39563346fb3f5ec",
		},
		wantSeqProfile:     1,
		wantBitDepth:       8,
		wantSubsamplingX:   false,
		wantSubsamplingY:   false,
		wantTileCols:       2,
		wantTileRows:       1,
		wantTileGroupTiles: 2,
	},
	{
		// Profile 1: 4:4:4 10-bit INTER. 8 frames (1 keyframe + 7 inter),
		// guarding high-bit-depth non-4:2:0 inter reconstruction.
		name: "profile1-444-10bit-inter-64x64",
		file: "profile1-444-10bit-inter-64x64.ivf",
		frameMD5Hex: []string{
			"3250b3e6c554a9725e372aba0e6f1836",
			"bbcef705eff277b6134c82d154dfd0e0",
			"5588133b22b316a0816a88bda01d711c",
			"fe22846dd23a8ea296132d0b299d772b",
			"d1b019af0ad85b194d474abe0698586b",
			"9d6cf7fd0eda963a93d553b7716e76b4",
			"5fb4e9e5ccee31c503b160de7a173c03",
			"ff7ee0fb4599801fc3015b9dd744ab1f",
		},
		wantSeqProfile:   1,
		wantBitDepth:     10,
		wantSubsamplingX: false,
		wantSubsamplingY: false,
	},
	{
		// Profile 1: 4:4:4 10-bit odd 130x130 INTER, guarding edge motion
		// prediction on high-bit-depth non-4:2:0 chroma.
		name: "profile1-444-10bit-edgemv-130x130",
		file: "profile1-444-10bit-edgemv-130x130.ivf",
		frameMD5Hex: []string{
			"56de8e3645276aa7322d4d26ddf4d048",
			"490eb309b57743705ad4a941b5d8dc89",
			"da2ff78bcd92690494532709c4d1e6a9",
			"f31fb72982c6141082dc7864e341f47a",
			"c03c86fb63b5a556b37f729b344a2efe",
		},
		wantSeqProfile:   1,
		wantBitDepth:     10,
		wantSubsamplingX: false,
		wantSubsamplingY: false,
		wantInterFrames:  1,
	},
	{
		// Profile 1: 4:4:4 10-bit inter multi-tile, combining high-bit-depth
		// non-subsampled chroma, reference frames, and split tile layout.
		name: "profile1-444-10bit-multitile-inter-2x1-128x128",
		file: "profile1-444-10bit-multitile-inter-2x1-128x128.ivf",
		frameMD5Hex: []string{
			"622e6159c052250bebf439858a73ec97",
			"5c3fd1c4289cb96721ec8ab01b26f039",
			"e60e744ca2d50f4701952b15fda80cfe",
			"cc6ee6ed27717c5341606e3d4371affb",
			"e31c6c9af577991ed02f043a0523ada8",
		},
		wantSeqProfile:     1,
		wantBitDepth:       10,
		wantSubsamplingX:   false,
		wantSubsamplingY:   false,
		wantInterFrames:    4,
		wantTileCols:       2,
		wantTileRows:       1,
		wantTileGroupTiles: 2,
	},
	{
		// Profile 2: 4:2:2 8-bit INTER. 8 frames (1 keyframe + 7 inter,
		// kf-max-dist=30) from a moving synthetic source.
		name: "profile2-422-8bit-inter-64x64",
		file: "profile2-422-8bit-inter-64x64.ivf",
		frameMD5Hex: []string{
			"e10a47c64fa7318c45ce0762bea3f0ce",
			"393c471479eff858fe4901cee94b6e57",
			"6251e9f45b4733552ba2b687b4c38154",
			"43fbaf637130f61feaaed7e8844c5d4b",
			"7ae17990bf39d7b507a441095ebe1ecb",
			"275861a447790a60e26cf3f574cab809",
			"61013488539b1b1a169f1e372d96d180",
			"0eca225d9dd9f6f768ab49a216511304",
		},
		wantSeqProfile:   2,
		wantBitDepth:     8,
		wantSubsamplingX: true,
		wantSubsamplingY: false,
	},
	{
		// Profile 2: 4:2:2 10-bit INTER with active CDEF and loop restoration,
		// guarding high-bit-depth professional-profile chroma through
		// inter-prediction and post-filter publication.
		name: "profile2-422-10bit-inter-cdef-restoration-160x128",
		file: "profile2-422-10bit-inter-cdef-restoration-160x128.ivf",
		frameMD5Hex: []string{
			"b5f242b59ee8ea0eb7d13df19d5d5ced",
			"a51c61951c1f3b2d9ee87a31c9c95020",
			"73cecc2864d5b2c36c4411d5dce7a93e",
			"eb98cdf9365718019e415e8793f35776",
			"87d2ad38f163244327bf9f5d3d29e161",
			"5673a85048021891382bfa66da1eb7d8",
			"cb55a9f1eff1491fd72186686d878ba8",
			"47f035eb1f1df9c34f43a50d919dae7e",
		},
		wantSeqProfile:        2,
		wantBitDepth:          10,
		wantSubsamplingX:      true,
		wantSubsamplingY:      false,
		wantCDEFFrames:        1,
		wantRestorationFrames: 1,
		wantInterFrames:       7,
	},
	{
		// Profile 2: 4:2:0 12-bit. 2 bytes/sample, BitDepth=12.
		name: "profile2-420-12bit-64x64",
		file: "profile2-420-12bit-64x64.ivf",
		frameMD5Hex: []string{
			"e714741e4ad4fce5a4469c79705f132c",
			"447103c7d7358e4cbb6f5b98ce4e1be1",
			"dcd86cbbf80f81d9699eaf6c6e879e72",
		},
		wantSeqProfile:   2,
		wantBitDepth:     12,
		wantSubsamplingX: true,
		wantSubsamplingY: true,
	},
	{
		// Profile 2: 4:2:0 12-bit odd 66x66 size with active CDEF, guarding
		// high-bit-depth chroma subsampling at frame edges that do not align
		// to a superblock or chroma-pair boundary.
		name: "profile2-420-12bit-66x66",
		file: "profile2-420-12bit-66x66.ivf",
		frameMD5Hex: []string{
			"f8dc71c2aafcf0ed77ce9d91a0a8ca0d",
			"8d44478c32115672846d4579d3962b58",
			"9825261dad9f100b51cba05ad534029d",
		},
		wantSeqProfile:   2,
		wantBitDepth:     12,
		wantSubsamplingX: true,
		wantSubsamplingY: true,
		wantCDEFFrames:   1,
	},
	{
		// Profile 2: 4:2:0 12-bit with active film grain, guarding
		// high-bit-depth grain synthesis on the professional-profile path.
		name: "profile2-420-12bit-filmgrain-96x96",
		file: "profile2-420-12bit-filmgrain-96x96.ivf",
		frameMD5Hex: []string{
			"48d28098e31d5ef96fa13d69bf13a374",
			"774b0173bb05d4d3d271d3e04303ac1b",
			"490cb07ef4b11d39480b11947796ef4b",
		},
		wantSeqProfile:      2,
		wantBitDepth:        12,
		wantSubsamplingX:    true,
		wantSubsamplingY:    true,
		wantFilmGrainFrames: 1,
	},
	{
		// Profile 2: 4:2:0 12-bit super-res. The coded width is smaller than the
		// displayed 160x128 output, guarding high-bit-depth super-res plumbing.
		name: "profile2-420-12bit-superres-160x128",
		file: "profile2-420-12bit-superres-160x128.ivf",
		frameMD5Hex: []string{
			"ea09f096227bac3e03277487354337fc",
			"2f144bf1f5144412cb606e79f8d74770",
			"92beb6d99acf1d73926dbed7b1eeb1ea",
			"17ae4f314b20aecc1d20a83bdee07da0",
		},
		wantSeqProfile:   2,
		wantBitDepth:     12,
		wantSubsamplingX: true,
		wantSubsamplingY: true,
		superRes:         true,
	},
	{
		// Profile 1: 4:4:4 8-bit with active CDEF and loop restoration.
		name: "profile1-444-8bit-cdef-restoration-160x128",
		file: "profile1-444-8bit-cdef-restoration-160x128.ivf",
		frameMD5Hex: []string{
			"c7001ffcb04bce20728f246f5494e58a",
			"f1a3ce1e10e2e84e1e61c46e735201ad",
			"1aa1aa327a503776ff58af351198037a",
			"7cfb09c54857fc6dd3994ef09e502fae",
		},
		wantSeqProfile:        1,
		wantBitDepth:          8,
		wantSubsamplingX:      false,
		wantSubsamplingY:      false,
		wantCDEFFrames:        1,
		wantRestorationFrames: 1,
	},
	{
		// Profile 1: 4:4:4 8-bit odd 130x130 size with active CDEF and
		// loop restoration, guarding byte-path non-4:2:0 edge filtering.
		name: "profile1-444-8bit-edge-cdef-restoration-130x130",
		file: "profile1-444-8bit-edge-cdef-restoration-130x130.ivf",
		frameMD5Hex: []string{
			"8d23e95555d1c482b0c9a9aaf1a82bd0",
			"ccd49de94b63207de8cbc935a5f63e5d",
			"deaa20627918286a339c1c6b30dea119",
		},
		wantSeqProfile:        1,
		wantBitDepth:          8,
		wantSubsamplingX:      false,
		wantSubsamplingY:      false,
		wantCDEFFrames:        1,
		wantRestorationFrames: 1,
	},
	{
		// Profile 1: 4:4:4 10-bit with active CDEF and loop restoration.
		name: "profile1-444-10bit-cdef-restoration-160x128",
		file: "profile1-444-10bit-cdef-restoration-160x128.ivf",
		frameMD5Hex: []string{
			"851bbe5c81da62f19d13b1efee5b60b5",
			"07da346eb6963d2bf2a7c37eb626b09e",
			"da669f9bf3e0a82727542a1f538b7226",
			"496024c0fd9b1708ca5a9255e317ebf0",
		},
		wantSeqProfile:        1,
		wantBitDepth:          10,
		wantSubsamplingX:      false,
		wantSubsamplingY:      false,
		wantCDEFFrames:        1,
		wantRestorationFrames: 1,
	},
	{
		// Profile 1: 4:4:4 10-bit odd 130x130 size with active CDEF and
		// loop restoration, guarding high-bit-depth non-4:2:0 edge handling.
		name: "profile1-444-10bit-edge-cdef-restoration-130x130",
		file: "profile1-444-10bit-edge-cdef-restoration-130x130.ivf",
		frameMD5Hex: []string{
			"3fedce7a89a109e6491429ea1b3aa2bc",
			"ff141f5d501503efeed8c32824ec51d4",
			"d996b27d544709b314685f1630599efe",
		},
		wantSeqProfile:        1,
		wantBitDepth:          10,
		wantSubsamplingX:      false,
		wantSubsamplingY:      false,
		wantCDEFFrames:        1,
		wantRestorationFrames: 1,
	},
	{
		// Profile 1: 4:4:4 8-bit with active film grain, guarding luma and
		// chroma grain synthesis on non-subsampled chroma output.
		name: "profile1-444-8bit-filmgrain-96x96",
		file: "profile1-444-8bit-filmgrain-96x96.ivf",
		frameMD5Hex: []string{
			"db1f78af3fbd46eec482cfcc02c59157",
			"f0abb8e97190db604aba5ffae2dd6b47",
			"706e3f8b36da82f717a1496cbbddf0c9",
		},
		wantSeqProfile:      1,
		wantBitDepth:        8,
		wantSubsamplingX:    false,
		wantSubsamplingY:    false,
		wantFilmGrainFrames: 1,
	},
	{
		// Profile 1: 4:4:4 10-bit with active film grain, guarding high-bit-depth
		// grain synthesis on non-subsampled chroma output.
		name: "profile1-444-10bit-filmgrain-96x96",
		file: "profile1-444-10bit-filmgrain-96x96.ivf",
		frameMD5Hex: []string{
			"fc7889878eb34c3fc3044b6353de35a6",
			"d058aaa10ea0893f8bd3d07a52f3ebdc",
			"4e59d8318286bb75836070f420d706a5",
		},
		wantSeqProfile:      1,
		wantBitDepth:        10,
		wantSubsamplingX:    false,
		wantSubsamplingY:    false,
		wantFilmGrainFrames: 1,
	},
	{
		// Profile 1: 4:4:4 8-bit super-res. 4 all-keyframes, super-res denom
		// 12, with CDEF and restoration disabled to isolate the upscale path.
		name: "profile1-444-8bit-superres-160x128",
		file: "profile1-444-8bit-superres-160x128.ivf",
		frameMD5Hex: []string{
			"315663b1b307e46d4d4719ad759b39c2",
			"b1c26bbbf277ac82c7fbdc0807974a1f",
			"bd6aad53bf088c7ff5f54d93c59ff649",
			"775085d7d08c8608b0acd2defa0fdc58",
		},
		wantSeqProfile:   1,
		wantBitDepth:     8,
		wantSubsamplingX: false,
		wantSubsamplingY: false,
		superRes:         true,
	},
	{
		// Profile 1: 4:4:4 8-bit super-res inter stream. Later frames reference
		// the previously upscaled output, guarding the super-res reference path.
		name: "profile1-444-8bit-superres-inter-160x128",
		file: "profile1-444-8bit-superres-inter-160x128.ivf",
		frameMD5Hex: []string{
			"7057484f2d2048053692a9bc41ce197b",
			"6b136fd484722654825fe6abc2e1773a",
			"cddb59775cd003ed4624b782fce4eca4",
			"32e3240e78dd8f53e7b673c9120fbe86",
			"5054adcd48bad5490911bb07248fea75",
			"150acaa0e5a682d454d7538b15cc4f2f",
			"5d0ab2de3f3e9ce25036226b11375e49",
			"78c4f82e984b44a272930333264bc764",
		},
		wantSeqProfile:   1,
		wantBitDepth:     8,
		wantSubsamplingX: false,
		wantSubsamplingY: false,
		wantInterFrames:  1,
		superRes:         true,
	},
	{
		// Profile 1: 4:4:4 10-bit super-res. 4 all-keyframes, super-res denom
		// 12, exercising the high-bit-depth 4:4:4 upscaled display path.
		name: "profile1-444-10bit-superres-160x128",
		file: "profile1-444-10bit-superres-160x128.ivf",
		frameMD5Hex: []string{
			"c556c65e1ceceb72778a36a5351b1d96",
			"6e66c5c97438754259295a95dde348a4",
			"cbf0f9d0584eaaf09538091edfb8f885",
			"f6f4c4213052d92f0ae700e793cad629",
		},
		wantSeqProfile:   1,
		wantBitDepth:     10,
		wantSubsamplingX: false,
		wantSubsamplingY: false,
		superRes:         true,
	},
	{
		// Profile 1: 4:4:4 10-bit super-res plus active loop restoration. This
		// guards the high-bit-depth non-subsampled boundary handoff between the
		// caller-owned super-res output and post-upscale restoration.
		name: "profile1-444-10bit-superres-restoration-160x128",
		file: "profile1-444-10bit-superres-restoration-160x128.ivf",
		frameMD5Hex: []string{
			"7946827a2d34b524ed92cc9a3b8504b8",
			"5c9585eecc0bff58859f42933505bdef",
			"92508e59e357059ea42166a6befead4f",
			"883da41a575466a10360f4b90710a339",
		},
		wantSeqProfile:        1,
		wantBitDepth:          10,
		wantSubsamplingX:      false,
		wantSubsamplingY:      false,
		wantRestorationFrames: 1,
		superRes:              true,
	},
	{
		// Profile 1: 4:4:4 10-bit static super-res inter stream with loop filter
		// disabled. Later frames reference high-bit-depth upscaled output,
		// guarding the scaled super-res reference publication path.
		name: "profile1-444-10bit-superres-inter-static-160x128",
		file: "profile1-444-10bit-superres-inter-static-160x128.ivf",
		frameMD5Hex: []string{
			"ab4284f9b59b7cfd81bdf5ab27d7e10b",
			"7a72c06659a4a041bbdcf65a7f45bb83",
			"7a72c06659a4a041bbdcf65a7f45bb83",
			"7a72c06659a4a041bbdcf65a7f45bb83",
			"7a72c06659a4a041bbdcf65a7f45bb83",
			"7a72c06659a4a041bbdcf65a7f45bb83",
			"7a72c06659a4a041bbdcf65a7f45bb83",
			"7a72c06659a4a041bbdcf65a7f45bb83",
		},
		wantSeqProfile:   1,
		wantBitDepth:     10,
		wantSubsamplingX: false,
		wantSubsamplingY: false,
		wantInterFrames:  1,
		superRes:         true,
	},
	{
		// Profile 1: 4:4:4 10-bit moving super-res inter stream. The larger
		// 16x16 chroma TXBs guard inter chroma tx_type clamping against the
		// active extended transform set.
		name: "profile1-444-10bit-superres-inter-simple-160x128",
		file: "profile1-444-10bit-superres-inter-simple-160x128.ivf",
		frameMD5Hex: []string{
			"0a0b14f62deee8bdcedbdd2c648c6396",
			"e7d1a6243d38d554df4f63b2a9d1beaf",
			"51bb916a676b953252f33e514dab46fd",
			"15a0daafab712254269a4ce27800c1c9",
			"8114f35db56233a52e75e367eb9f7a33",
			"b5dfcdadffe2c9d246ad9930e39ab290",
			"36278c9c6a7e15d325a45887e11ab60a",
			"8e42428b236d37ca8d59c5c5e4ece89e",
		},
		wantSeqProfile:   1,
		wantBitDepth:     10,
		wantSubsamplingX: false,
		wantSubsamplingY: false,
		wantInterFrames:  7,
		superRes:         true,
	},
	{
		// Super-res only (no loop restoration, no CDEF). 4:2:0 8-bit, 160x128,
		// 4 all-keyframes, super-res denom 12 => coded width 107 (not a multiple
		// of 8), exercising the MI-aligned upscale source span. Guards the
		// super-res driver's aligned-width source fix.
		name: "superres-420-8bit-160x128",
		file: "superres-420-8bit-160x128.ivf",
		frameMD5Hex: []string{
			"024e5d1f55340eee39893567e8000554",
			"80c239661f2cd1afe5fce5038ceac76f",
			"93dc5cde59cb99bd524264b15c70a700",
			"87887afdc06ba5f0e88b2acbe1ce0ce5",
		},
		wantBitDepth:     8,
		wantSubsamplingX: true,
		wantSubsamplingY: true,
		superRes:         true,
	},
	{
		// Super-res + loop restoration. 4:2:0 8-bit, 160x128, 4 all-keyframes,
		// super-res denom 12. Restoration runs at the upscaled resolution and
		// consumes deblock boundary rows that are super-res-upscaled on the fly.
		// Guards the loop-restoration + super-res boundary-save path.
		name: "superres-restoration-420-8bit-160x128",
		file: "superres-restoration-420-8bit-160x128.ivf",
		frameMD5Hex: []string{
			"93533110e2bd60fed6f15b5e65178bc6",
			"bf83a513a868584aec7f9f05c5187d70",
			"4e7dc4a0d7ab9cebdbdfdf3ef73b0d1c",
			"6af9472b0679093facbefa85e973a7f9",
		},
		wantBitDepth:     8,
		wantSubsamplingX: true,
		wantSubsamplingY: true,
		superRes:         true,
	},
	{
		// Profile 0: 4:2:0 8-bit with use_128x128_superblock=true and a
		// forced 128x128 partition root, guarding the large-superblock decode
		// path against libaom output.
		name: "superblock128-420-8bit-128x128",
		file: "superblock128-420-8bit-128x128.ivf",
		frameMD5Hex: []string{
			"42ace8044951a222256ffcd65aa8bb67",
			"c1b26993930b06dec4f0d05457035565",
			"4e7b4292d96e53cc8ea5b16c8d7e13de",
		},
		wantBitDepth:      8,
		wantSubsamplingX:  true,
		wantSubsamplingY:  true,
		checkUse128x128SB: true,
		wantUse128x128SB:  true,
	},
	{
		name: "profile0-420-8bit-edgemv-130x130",
		file: "profile0-420-8bit-edgemv-130x130.ivf",
		frameMD5Hex: []string{
			"f6c673ec13ccdd994fd10373a1d7f814",
			"a3b01764f4b3e9c7bcec7c0e7e86be64",
			"8158c19b78deb9ac6aaab9d411fe268f",
			"eebdcf491352f1b13a24b614ba137832",
			"54582e43045ece242c0a50e3f73e7a6b",
		},
		wantBitDepth:     8,
		wantSubsamplingX: true,
		wantSubsamplingY: true,
	},
	{
		name: "profile2-420-12bit-edgemv-130x130",
		file: "profile2-420-12bit-edgemv-130x130.ivf",
		frameMD5Hex: []string{
			"f85baff9f8f516879ab8da538e582a09",
			"5dfd30b2a7eb7e2178a06bd7e060bc80",
			"f14ff5bc1a1504908f1d22b4be5a0ca3",
			"24e96db2d3644a748e587dd011863a4a",
			"325a954def1c43d852c980ddc66ba652",
		},
		wantSeqProfile:   2,
		wantBitDepth:     12,
		wantSubsamplingX: true,
		wantSubsamplingY: true,
	},
	{
		name: "multitile-2x1-rows-256x256",
		file: "multitile-2x1-rows-256x256.ivf",
		frameMD5Hex: []string{
			"19abff965e7c0b982b779b14ce5d67a9",
			"d9670540a9ef1cc444eb684dde65928c",
			"1ece7ab7594b417ad313c395c9bf153c",
			"6de64107acac0d89ad361dfe615a1c5f",
		},
		wantBitDepth:     8,
		wantSubsamplingX: true,
		wantSubsamplingY: true,
	},
}

// TestProfileVendoredClips decodes each vendored profile-conformance clip
// through the byte-exact framework pipeline and asserts every visible frame
// matches its libaom aomdec golden MD5. Each clip gets a brand-new decoder
// with fully local state.
func TestProfileVendoredClips(t *testing.T) {
	for _, clip := range profileClips {
		clip := clip
		t.Run(clip.name, func(t *testing.T) {
			runProfileClip(t, clip)
		})
	}
}

func runProfileClip(t *testing.T, clip profileClip) {
	t.Helper()

	wantDigests := make([]testvector.MD5, len(clip.frameMD5Hex))
	for i, hexStr := range clip.frameMD5Hex {
		md5, err := testvector.ParseMD5Hex([]byte(hexStr))
		if err != nil {
			t.Fatalf("parse golden md5[%d]=%q: %v", i, hexStr, err)
		}
		wantDigests[i] = md5
	}

	ivfData := readProfileClip(t, clip.file)
	it, err := ivf.NewIterator(ivfData)
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	// All decoder state below is local to this call; nothing is shared with
	// any other clip, test, or package.
	var stream decoder.Stream
	var refs decoder.SurfaceReferences
	var state decoder.FrameWorkState
	var pool frame.Pool
	var backing []byte
	var frameSlots []frame.Frame
	var free []int
	var used []bool
	var outputPool frame.Pool
	var outputBacking []byte
	var outputFrameSlots []frame.Frame
	var outputFree []int
	var outputUsed []bool
	var frameContexts decoder.SharedFrameContextStore
	var havePool bool
	var mvEntryBacking []tile.ReferenceMVEntry
	var temporalEntryBacking []tile.TemporalMotionEntry
	var mvFrames []tile.ReferenceMVFrame
	var mvStore []tile.TemporalMotionReferenceFrame
	var mvLength int
	var publishedMVEntryBacking []tile.ReferenceMVEntry
	var publishedMVFrames []tile.ReferenceMVFrame

	var events [16]decoder.Event
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [parser.MaxTiles]parser.TileSpan
	var jobs [parser.MaxTiles]tile.Job
	var batches [parser.MaxTiles]threading.Batch
	var releases [parser.RefFrames]int

	// keepAlive prevents the backing arenas from being GC'd while the pool's
	// *frame.Frame pointers are live across iterations.
	defer func() {
		runtime.KeepAlive(backing)
		runtime.KeepAlive(frameSlots)
		runtime.KeepAlive(free)
		runtime.KeepAlive(used)
		runtime.KeepAlive(outputBacking)
		runtime.KeepAlive(outputFrameSlots)
		runtime.KeepAlive(outputFree)
		runtime.KeepAlive(outputUsed)
		runtime.KeepAlive(mvEntryBacking)
		runtime.KeepAlive(temporalEntryBacking)
		runtime.KeepAlive(mvFrames)
		runtime.KeepAlive(mvStore)
		runtime.KeepAlive(publishedMVEntryBacking)
		runtime.KeepAlive(publishedMVFrames)
	}()

	emitted := 0
	paletteYBlocks := 0
	paletteUVBlocks := 0
	cdefFrames := 0
	restorationFrames := 0
	filmGrainFrames := 0
	interFrames := 0
	switchFrames := 0
	maxTileCols := 0
	maxTileRows := 0
	maxTileGroupTiles := 0
	checkedColorConfig := false
	for {
		ivfFrame, ok, err := it.Next()
		if err != nil {
			t.Fatalf("ivf frame %d: %v", emitted, err)
		}
		if !ok {
			break
		}
		count, err := stream.PushLowOverhead(ivfFrame.Payload, events[:])
		if err != nil {
			t.Fatalf("ivf frame %d PushLowOverhead: %v", ivfFrame.Index, err)
		}
		for i := 0; i < count; i++ {
			event := events[i]
			if !eventRunsFrameWork(event) {
				continue
			}
			if cols := int(event.TileInfo.Cols); cols > maxTileCols {
				maxTileCols = cols
			}
			if rows := int(event.TileInfo.Rows); rows > maxTileRows {
				maxTileRows = rows
			}
			if tiles := int(event.TileGroup.TileCount); tiles > maxTileGroupTiles {
				maxTileGroupTiles = tiles
			}
			if !checkedColorConfig {
				if event.SequenceHeader.SeqProfile != clip.wantSeqProfile {
					t.Fatalf("SeqProfile=%d want %d", event.SequenceHeader.SeqProfile, clip.wantSeqProfile)
				}
				cc := event.SequenceHeader.ColorConfig
				if cc.BitDepth != clip.wantBitDepth {
					t.Fatalf("BitDepth=%d want %d", cc.BitDepth, clip.wantBitDepth)
				}
				if cc.MonoChrome {
					t.Fatalf("MonoChrome=true want false")
				}
				if cc.SubsamplingX != clip.wantSubsamplingX || cc.SubsamplingY != clip.wantSubsamplingY {
					t.Fatalf("subsampling=(%v,%v) want (%v,%v)",
						cc.SubsamplingX, cc.SubsamplingY, clip.wantSubsamplingX, clip.wantSubsamplingY)
				}
				if clip.checkUse128x128SB && event.SequenceHeader.Use128x128Superblock != clip.wantUse128x128SB {
					t.Fatalf("Use128x128Superblock=%v want %v",
						event.SequenceHeader.Use128x128Superblock, clip.wantUse128x128SB)
				}
				checkedColorConfig = true
			}
			if !havePool {
				pool, backing, frameSlots, free, used = bindFramePool(t, event, 8)
				mvEntryBacking, temporalEntryBacking, mvFrames, mvStore, mvLength = bindMotionStore(t, event, 8)
				if clip.superRes {
					outputPool, outputBacking, outputFrameSlots, outputFree, outputUsed = bindOutputFramePool(t, event, 8)
					mvStore = make([]tile.TemporalMotionReferenceFrame, profileSuperResOutputGlobalBase+8)
					publishedMVEntryBacking = make([]tile.ReferenceMVEntry, 8*mvLength)
					publishedMVFrames = make([]tile.ReferenceMVFrame, 8)
				}
				havePool = true
			}

			var postMD5 testvector.MD5
			havePost := false
			currentMVSurface := -1
			tileRunner := batchRunner(func(ctx decoder.FrameWorkBatch) error {
				surface, err := ctx.Surface()
				if err != nil {
					return err
				}
				if currentMVSurface != surface || mvFrames[surface].Entries == nil {
					first := surface * mvLength
					currentMVFrame, err := ctx.BindReferenceMVFrame(mvEntryBacking[first : first+mvLength])
					if err != nil {
						return err
					}
					mvFrames[surface] = currentMVFrame
					currentMVSurface = surface
				}
				ctx.CurrentMVFrame = &mvFrames[surface]
				if ctx.TileInfo.UseRefFrameMVS {
					temporalMVs, err := ctx.BindTemporalMotionField(temporalEntryBacking)
					if err != nil {
						return err
					}
					ctx.TemporalMVs = &temporalMVs
					resolved, err := decoder.ResolveTemporalMotionReferences(referenceSurfaces[:len(ctx.References)], mvStore, ctx.ReferenceMVs[:])
					if err != nil {
						return err
					}
					if resolved != len(ctx.References) {
						return decoder.ErrInvalidSurfaceReference
					}
					if _, err := ctx.SetupTemporalMotionField(); err != nil {
						return err
					}
				}
				var restorationReq *threading.FrameWorkTileRestorationRequest
				if ctx.RestorationFrameBuffers != nil {
					restoration := threading.FrameWorkTileRestorationRequest{Buffers: *ctx.RestorationFrameBuffers}
					if err := restoration.InitReferences(); err != nil {
						return err
					}
					restorationReq = &restoration
				}
				for j := 0; j < len(ctx.Jobs); j++ {
					var decodeState tile.DecodeState
					if err := ctx.JobDecodeState(j, &decodeState); err != nil {
						return err
					}
					var storage threading.FrameWorkTileResidualCDFStorage
					if err := ctx.InitTileResidualCDFStorage(&storage); err != nil {
						return err
					}
					var scratch threading.FrameWorkTileResidualScratch
					rootCols, err := ctx.JobBlockLoopContextRootColumns(j)
					if err != nil {
						return err
					}
					scratch.LoopContext.Above = make([]tile.BlockLoopRootAboveContext, rootCols)
					loopReq, err := ctx.JobBlockLoopRequest(j, nil, nil, 0)
					if err != nil {
						return err
					}
					loopReq.DecodePredictionModes = true
					loopReq.DecodeInterModes = true
					loopReq.DecodeMotionVectors = true
					loopReq.DecodeInterIntra = true
					loopReq.DecodeMotionModes = true
					loopReq.DecodeCompoundBlend = true
					int32Scratch, residualScratch, err := residualScratch(ctx)
					if err != nil {
						return err
					}
					var interScratch threading.FrameWorkInterPredictionScratch
					predictionScratch := threading.FrameWorkPredictionScratch{Inter: &interScratch}
					_, err = ctx.DecodeAndReconstructJobResiduals(j, &decodeState, storage.CDFs(), &scratch, threading.FrameWorkTileResidualRequest{
						Loop:          loopReq,
						TransformMode: ctx.TransformRef.TransformMode,
						CDEFIndexMap:  ctx.CDEFIndexMap,
						LoopFilterMap: ctx.LoopFilterMap,
						Restoration:   restorationReq,
						Transforms: func(visit tile.BlockLoopVisit) (threading.FrameWorkBlockTransforms, error) {
							if visit.Prediction.Valid && !visit.Prediction.Intra {
								return ctx.ReadInterBlockTransforms(&decodeState, visit)
							}
							return ctx.ReadIntraBlockTransforms(&decodeState, visit)
						},
						AfterBlock: func(visit tile.BlockLoopVisit) error {
							if visit.Prediction.Palette.YSize > 0 {
								paletteYBlocks++
							}
							if visit.Prediction.Palette.UVSize > 0 {
								paletteUVBlocks++
							}
							return nil
						},
						Int32Scratch:      int32Scratch,
						ResidualScratch:   residualScratch,
						PredictionScratch: &predictionScratch,
					})
					if err != nil {
						return fmt.Errorf("decode/reconstruct job %d: %w", j, err)
					}
					if err := ctx.RetainTileResidualCDFStorage(j, &decodeState, &storage); err != nil {
						return err
					}
				}
				return nil
			})
			post := &profilePostFilterRunner{fn: func(ctx decoder.FrameWorkPostFilterContext) (int, error) {
				active := ctx.ActivePostFilters()
				if active.Has(decoder.FrameWorkPostFilterCDEF) {
					cdefFrames++
				}
				if active.Has(decoder.FrameWorkPostFilterLoopRestoration) {
					restorationFrames++
				}
				if active.Has(decoder.FrameWorkPostFilterFilmGrain) {
					filmGrainFrames++
				}
				if clip.superRes {
					out, publishedGlobalSurface, ev, err := applyCallerPostFiltersForPublication(ctx, &outputPool)
					if err != nil {
						return -1, err
					}
					published := false
					defer func() {
						if !published {
							_ = releaseProfileSuperResOutput(&outputPool, publishedGlobalSurface)
						}
					}()
					if currentMVSurface >= 0 {
						if err := publishProfileSuperResTemporalMotionReference(ev, publishedGlobalSurface, &mvFrames[currentMVSurface], publishedMVFrames, publishedMVEntryBacking, mvStore, mvLength); err != nil {
							return -1, err
						}
					}
					got, err := testvector.FrameMD5(*out)
					if err != nil {
						return -1, err
					}
					postMD5 = got
					havePost = true
					published = true
					return publishedGlobalSurface, nil
				}
				post := decoder.FrameWorkBoundSupportedPostFilterRunner{}
				size, err := supportedPostFilterScratchLen(ctx)
				if err != nil {
					return -1, err
				}
				post.Scratch = postFilterScratchStorage(size)
				if err := post.Apply(ctx); err != nil {
					return -1, err
				}
				if currentMVSurface >= 0 {
					if err := decoder.PublishTemporalMotionReference(post.Context.Event, currentMVSurface, &mvFrames[currentMVSurface], mvStore); err != nil {
						return -1, err
					}
				}
				output := post.Context.Output
				if post.DisplayOutput != nil {
					output = post.DisplayOutput
				}
				got, err := testvector.FrameMD5(*output)
				if err != nil {
					return -1, err
				}
				postMD5 = got
				havePost = true
				return -1, nil
			}}
			var result decoder.FrameWorkEventResult
			var err error
			if clip.superRes {
				provider := profileSurfaceProvider{coded: &pool, output: &outputPool}
				result, err = state.RunEventWithContextAndExternalReferences(&refs, &pool, event.SequenceHeader, event, 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, provider, func(local int) int { return local }, provider, &frameContexts, sideDataRunner{}, tileRunner, post)
			} else {
				result, err = state.RunEventWithContextAndSideDataAndPostFilterRunners(&refs, &pool, event.SequenceHeader, event, 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, sideDataRunner{}, tileRunner, post)
			}
			if err != nil {
				t.Fatalf("ivf frame %d run framework event: %v", ivfFrame.Index, err)
			}
			if !result.Run.CompletedFrame {
				continue
			}
			if event.FrameHeader.FrameType == parser.FrameTypeSwitch {
				switchFrames++
			}
			if !event.FrameHeader.ShowFrame && !clip.includeHiddenFrameMD5 {
				continue
			}
			if event.FrameHeader.FrameType == parser.FrameTypeInter {
				interFrames++
			}
			if !havePost {
				t.Fatalf("ivf frame %d completed without postfilter", ivfFrame.Index)
			}
			if emitted >= len(wantDigests) {
				t.Fatalf("ivf frame %d: more visible frames than golden digests (%d)", ivfFrame.Index, len(wantDigests))
			}
			if postMD5 != wantDigests[emitted] {
				t.Fatalf("frame %d md5 got=%x want=%x (libaom golden)", emitted, postMD5, wantDigests[emitted])
			}
			emitted++
		}
	}
	if emitted != len(wantDigests) {
		t.Fatalf("emitted %d visible frames, want %d golden digests", emitted, len(wantDigests))
	}
	if paletteYBlocks < clip.wantPaletteYBlocks {
		t.Fatalf("palette y blocks=%d want at least %d", paletteYBlocks, clip.wantPaletteYBlocks)
	}
	if paletteUVBlocks < clip.wantPaletteUVBlocks {
		t.Fatalf("palette uv blocks=%d want at least %d", paletteUVBlocks, clip.wantPaletteUVBlocks)
	}
	if cdefFrames < clip.wantCDEFFrames {
		t.Fatalf("cdef frames=%d want at least %d", cdefFrames, clip.wantCDEFFrames)
	}
	if restorationFrames < clip.wantRestorationFrames {
		t.Fatalf("restoration frames=%d want at least %d", restorationFrames, clip.wantRestorationFrames)
	}
	if filmGrainFrames < clip.wantFilmGrainFrames {
		t.Fatalf("film grain frames=%d want at least %d", filmGrainFrames, clip.wantFilmGrainFrames)
	}
	if interFrames < clip.wantInterFrames {
		t.Fatalf("inter frames=%d want at least %d", interFrames, clip.wantInterFrames)
	}
	if switchFrames < clip.wantSwitchFrames {
		t.Fatalf("switch frames=%d want at least %d", switchFrames, clip.wantSwitchFrames)
	}
	if clip.wantTileCols != 0 && maxTileCols != clip.wantTileCols {
		t.Fatalf("tile cols=%d want %d", maxTileCols, clip.wantTileCols)
	}
	if clip.wantTileRows != 0 && maxTileRows != clip.wantTileRows {
		t.Fatalf("tile rows=%d want %d", maxTileRows, clip.wantTileRows)
	}
	if clip.wantTileGroupTiles != 0 && maxTileGroupTiles < clip.wantTileGroupTiles {
		t.Fatalf("tile group tiles=%d want at least %d", maxTileGroupTiles, clip.wantTileGroupTiles)
	}
}

func readProfileClip(t *testing.T, name string) []byte {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	path := filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", "testdata", "profiles", name))
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read clip %s: %v", path, err)
	}
	return data
}

func eventRunsFrameWork(event decoder.Event) bool {
	switch event.Kind {
	case decoder.EventFrameHeader, decoder.EventFrame, decoder.EventTileGroup, decoder.EventExistingFrame:
		return true
	default:
		return false
	}
}

func bindFramePool(t *testing.T, event decoder.Event, count int) (frame.Pool, []byte, []frame.Frame, []int, []bool) {
	t.Helper()
	format := frameFormatFromEvent(event)
	layout, err := frame.RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*count)
	frames := make([]frame.Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	pool, err := frame.BindPool(backing, format, frames, free, used)
	if err != nil {
		t.Fatal(err)
	}
	return pool, backing, frames, free, used
}

func bindOutputFramePool(t *testing.T, event decoder.Event, count int) (frame.Pool, []byte, []frame.Frame, []int, []bool) {
	t.Helper()
	format := outputFrameFormatFromEvent(event)
	layout, err := frame.RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*count)
	frames := make([]frame.Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	pool, err := frame.BindPool(backing, format, frames, free, used)
	if err != nil {
		t.Fatal(err)
	}
	return pool, backing, frames, free, used
}

func bindMotionStore(t *testing.T, event decoder.Event, count int) ([]tile.ReferenceMVEntry, []tile.TemporalMotionEntry, []tile.ReferenceMVFrame, []tile.TemporalMotionReferenceFrame, int) {
	t.Helper()
	miCols := miExtent(event.FrameSize.CodedWidth)
	miRows := miExtent(event.FrameSize.Height)
	length, err := tile.ReferenceMVFrameEntries(miRows, miCols)
	if err != nil {
		t.Fatal(err)
	}
	return make([]tile.ReferenceMVEntry, count*length),
		make([]tile.TemporalMotionEntry, length),
		make([]tile.ReferenceMVFrame, count),
		make([]tile.TemporalMotionReferenceFrame, count),
		length
}

func miExtent(pixels uint32) uint32 {
	return ((pixels + 7) >> 3) << 1
}

func frameFormatFromEvent(event decoder.Event) frame.Format {
	return frameFormatFromEventWidth(event, event.FrameSize.CodedWidth)
}

func outputFrameFormatFromEvent(event decoder.Event) frame.Format {
	width := event.FrameSize.UpscaledWidth
	if width == 0 {
		width = event.FrameSize.CodedWidth
	}
	return frameFormatFromEventWidth(event, width)
}

func frameFormatFromEventWidth(event decoder.Event, width uint32) frame.Format {
	sbSizeLog2 := uint8(6)
	if event.SequenceHeader.Use128x128Superblock {
		sbSizeLog2 = 7
	}
	return frame.Format{
		Width:        int(width),
		Height:       int(event.FrameSize.Height),
		BitDepth:     event.SequenceHeader.ColorConfig.BitDepth,
		MonoChrome:   event.SequenceHeader.ColorConfig.MonoChrome,
		SubsamplingX: event.SequenceHeader.ColorConfig.SubsamplingX,
		SubsamplingY: event.SequenceHeader.ColorConfig.SubsamplingY,
		SBSizeLog2:   sbSizeLog2,
		Align:        32,
	}
}

type profileSurfaceProvider struct {
	coded  *frame.Pool
	output *frame.Pool
}

func (p profileSurfaceProvider) FrameSurface(id int) (*frame.Frame, error) {
	pool, local, err := p.resolve(id)
	if err != nil {
		return nil, err
	}
	return pool.Frame(local)
}

func (p profileSurfaceProvider) ReleaseFrameSurfaces(ids []int) error {
	if len(ids) == 0 {
		return nil
	}
	for i, id := range ids {
		pool, local, err := p.resolve(id)
		if err != nil {
			return err
		}
		if _, err := pool.Frame(local); err != nil {
			return err
		}
		for j := range i {
			if ids[j] == id {
				return frame.ErrInvalidSlot
			}
		}
	}
	for _, id := range ids {
		pool, local, err := p.resolve(id)
		if err != nil {
			return err
		}
		if err := pool.Release(local); err != nil {
			return err
		}
	}
	return nil
}

func (p profileSurfaceProvider) resolve(id int) (*frame.Pool, int, error) {
	if id < 0 {
		return nil, -1, decoder.ErrInvalidSurfaceReference
	}
	if id >= profileSuperResOutputGlobalBase {
		if p.output == nil {
			return nil, -1, decoder.ErrInvalidSurfaceReference
		}
		return p.output, id - profileSuperResOutputGlobalBase, nil
	}
	if p.coded == nil {
		return nil, -1, decoder.ErrInvalidSurfaceReference
	}
	return p.coded, id, nil
}

func publishProfileSuperResTemporalMotionReference(event decoder.Event, globalSurface int, current *tile.ReferenceMVFrame, published []tile.ReferenceMVFrame, backing []tile.ReferenceMVEntry, store []tile.TemporalMotionReferenceFrame, mvLength int) error {
	local := globalSurface - profileSuperResOutputGlobalBase
	if local < 0 || local >= len(published) || mvLength <= 0 {
		return decoder.ErrInvalidSurfaceReference
	}
	if len(backing) < (local+1)*mvLength {
		return decoder.ErrSurfaceReferenceBufferTooSmall
	}
	if current == nil || len(current.Entries) != mvLength {
		return decoder.ErrInvalidSurfaceReference
	}
	dst := backing[local*mvLength : (local+1)*mvLength]
	copy(dst, current.Entries)
	published[local] = tile.ReferenceMVFrame{
		Rows:    current.Rows,
		Cols:    current.Cols,
		Stride:  current.Stride,
		Entries: dst,
	}
	return decoder.PublishTemporalMotionReference(event, globalSurface, &published[local], store)
}

func residualScratch(ctx decoder.FrameWorkBatch) ([]int32, []int16, error) {
	q, _, err := ctx.BlockQuantizer(ctx.Quantization.BaseQIdx, 0, threading.FrameWorkPlaneY)
	if err != nil {
		return nil, nil, err
	}
	int32Len, int16Len, err := reconstruct.ScratchLen(reconstruct.Block{
		Size:      transform.Size{Width: 64, Height: 64},
		Transform: transform.TypeDCTDCT,
		Quantizer: q,
		Lossless:  false,
	})
	if err != nil {
		return nil, nil, err
	}
	return make([]int32, int32Len), make([]int16, int16Len), nil
}

type batchRunner func(decoder.FrameWorkBatch) error

func (fn batchRunner) Run(ctx decoder.FrameWorkBatch) error { return fn(ctx) }

type postFilterRunner func(decoder.FrameWorkPostFilterContext) error

func (fn postFilterRunner) Apply(ctx decoder.FrameWorkPostFilterContext) error { return fn(ctx) }

type profilePostFilterRunner struct {
	fn                     func(decoder.FrameWorkPostFilterContext) (int, error)
	publishedGlobalSurface int
}

func (r *profilePostFilterRunner) Apply(ctx decoder.FrameWorkPostFilterContext) error {
	if r == nil || r.fn == nil {
		return decoder.ErrInvalidFrameWorkState
	}
	r.publishedGlobalSurface = -1
	publishedGlobalSurface, err := r.fn(ctx)
	if err != nil {
		return err
	}
	r.publishedGlobalSurface = publishedGlobalSurface
	return nil
}

func (r *profilePostFilterRunner) PublishedFrameWorkGlobalSurface() (int, bool) {
	if r == nil || r.publishedGlobalSurface < 0 {
		return -1, false
	}
	return r.publishedGlobalSurface, true
}

type sideDataRunner struct{}

func (sideDataRunner) BindFrameWorkSideData(state *decoder.FrameWorkState, ctx decoder.FrameWorkBatch) error {
	size, err := decoder.FrameWorkSideDataScratchLen(ctx)
	if err != nil {
		return err
	}
	side, err := size.BindRunner(sideDataScratch(size))
	if err != nil {
		return err
	}
	return side.BindFrameWorkSideData(state, ctx)
}

func sideDataScratch(size decoder.FrameWorkSideDataScratchSize) decoder.FrameWorkSideDataScratch {
	return decoder.FrameWorkSideDataScratch{
		CDEFIndex:          make([]uint8, maxInt(size.CDEF, 0)),
		CDEFRead:           make([]bool, maxInt(size.CDEF, 0)),
		LoopFilterRecords:  make([]threading.FrameWorkLoopFilterBlockRecord, maxInt(size.LoopFilterRecords, 0)),
		RestorationRecords: make([]tile.RestorationUnitRecord, maxInt(size.RestorationRecords, 0)),
		RestorationAbove:   make([]uint16, maxInt(size.RestorationBoundary, 0)),
		RestorationBelow:   make([]uint16, maxInt(size.RestorationBoundary, 0)),
	}
}

// applyCallerPostFilters runs the caller-owned full post-filter chain
// (ApplyCallerPostFilters) for a super-res frame: loop filter, CDEF, super-res
// upscale into caller-owned scratch, then loop restoration and film grain at
// the upscaled resolution. It performs the two-pass super-res scratch sizing
// (size the upscale output frame first, then re-size for the post-super-res
// tail) the public caller runner uses, and returns the final display frame.
func applyCallerPostFilters(ctx decoder.FrameWorkPostFilterContext) (*frame.Frame, decoder.Event, error) {
	return applyCallerPostFiltersWithSuperResOutput(ctx, nil, nil)
}

func applyCallerPostFiltersForPublication(ctx decoder.FrameWorkPostFilterContext, outputPool *frame.Pool) (*frame.Frame, int, decoder.Event, error) {
	local, outputFrame, err := acquireProfileOutputSurface(outputPool, ctx.Event)
	if err != nil {
		return nil, -1, decoder.Event{}, err
	}
	published := false
	defer func() {
		if !published {
			_ = outputPool.Release(local)
		}
	}()
	backing, err := profileFrameBacking(outputFrame)
	if err != nil {
		return nil, -1, decoder.Event{}, err
	}
	out, ev, err := applyCallerPostFiltersWithSuperResOutput(ctx, backing, outputFrame)
	if err != nil {
		return nil, -1, decoder.Event{}, err
	}
	published = true
	return out, profileSuperResOutputGlobalBase + local, ev, nil
}

func releaseProfileSuperResOutput(outputPool *frame.Pool, globalSurface int) error {
	if globalSurface < profileSuperResOutputGlobalBase {
		return decoder.ErrInvalidSurfaceReference
	}
	return outputPool.Release(globalSurface - profileSuperResOutputGlobalBase)
}

func acquireProfileOutputSurface(outputPool *frame.Pool, event decoder.Event) (int, *frame.Frame, error) {
	if outputPool == nil {
		return -1, nil, frame.ErrInvalidPool
	}
	return outputPool.AcquireFormat(outputFrameFormatFromEvent(event))
}

func profileFrameBacking(outputFrame *frame.Frame) ([]byte, error) {
	if outputFrame == nil {
		return nil, frame.ErrInvalidSlot
	}
	if outputFrame.Layout.Size < 0 || cap(outputFrame.Y.Pix) < outputFrame.Layout.Size {
		return nil, frame.ErrShortBuffer
	}
	return outputFrame.Y.Pix[:outputFrame.Layout.Size], nil
}

func applyCallerPostFiltersWithSuperResOutput(ctx decoder.FrameWorkPostFilterContext, superResOutputFrame []byte, outputView *frame.Frame) (*frame.Frame, decoder.Event, error) {
	var scratchOutputView frame.Frame
	if outputView == nil {
		outputView = &scratchOutputView
	}
	options := decoder.FrameWorkPostFilterBindOptions{SuperResOutputView: outputView}
	if ctx.LoopFilterMap != nil {
		options.LoopFilterMap = *ctx.LoopFilterMap
	}
	if ctx.CDEFIndexMap != nil {
		options.CDEFIndexMap = *ctx.CDEFIndexMap
	}
	if ctx.RestorationFrameBuffers != nil {
		options.RestorationRecords = ctx.RestorationFrameBuffers.Records
		options.RestorationBoundaries = ctx.RestorationFrameBuffers.Boundaries
	}

	probe := callerProbeRequest(options)
	first, err := ctx.CallerPostFilterScratchLen(probe)
	if err != nil {
		return nil, decoder.Event{}, err
	}
	scratch := postFilterScratchStorage(first)
	probe = callerProbeRequest(options)
	probe.SuperRes.OutputFrame = scratch.SuperResOutputFrame
	if superResOutputFrame != nil {
		probe.SuperRes.OutputFrame = superResOutputFrame
	}
	probe.SuperRes.OutputView = outputView
	full, err := ctx.CallerPostFilterScratchLen(probe)
	if err != nil {
		return nil, decoder.Event{}, err
	}
	scratch = postFilterScratchStorage(full)
	if superResOutputFrame != nil {
		scratch.SuperResOutputFrame = superResOutputFrame
	}

	req, err := full.BindRequest(options, scratch)
	if err != nil {
		return nil, decoder.Event{}, err
	}
	runner := decoder.FrameWorkCallerPostFilterRunner{Request: req}
	if err := runner.Apply(ctx); err != nil {
		return nil, decoder.Event{}, err
	}
	if runner.Context.Output == nil {
		return nil, decoder.Event{}, fmt.Errorf("caller post-filter produced nil output")
	}
	return runner.Context.Output, runner.Context.Event, nil
}

func callerProbeRequest(options decoder.FrameWorkPostFilterBindOptions) decoder.FrameWorkPostFilterRequest {
	return decoder.FrameWorkPostFilterRequest{
		LoopFilter:  decoder.FrameWorkLoopFilterPostFilterRequest{Map: options.LoopFilterMap},
		CDEF:        decoder.FrameWorkCDEFPostFilterRequest{IndexMap: options.CDEFIndexMap},
		Restoration: decoder.FrameWorkRestorationPostFilterRequest{Records: options.RestorationRecords},
	}
}

func supportedPostFilterScratchLen(ctx decoder.FrameWorkPostFilterContext) (decoder.FrameWorkPostFilterScratchSize, error) {
	var probe decoder.FrameWorkPostFilterRequest
	if ctx.LoopFilterMap != nil {
		probe.LoopFilter.Map = *ctx.LoopFilterMap
	}
	if ctx.CDEFIndexMap != nil {
		probe.CDEF.IndexMap = *ctx.CDEFIndexMap
	}
	if ctx.RestorationFrameBuffers != nil {
		probe.Restoration.Records = ctx.RestorationFrameBuffers.Records
	}
	return ctx.SupportedPostFilterScratchLen(probe)
}

func postFilterScratchStorage(size decoder.FrameWorkPostFilterScratchSize) decoder.FrameWorkPostFilterScratch {
	scratch := decoder.FrameWorkPostFilterScratch{
		LoopFilterEdges:    make([]decoder.FrameWorkLoopFilterPostFilterEdge, maxInt(size.LoopFilter.Edges, 0)),
		LoopFilterSchedule: make([]uint32, maxInt(size.LoopFilter.Schedule, 0)),

		CDEFDirectionGrid: make([]cdef.DirectionGrid, maxInt(size.CDEF.DirectionGrid, 0)),
		CDEFVarianceGrid:  make([]cdef.VarianceGrid, maxInt(size.CDEF.VarianceGrid, 0)),
		CDEFInput:         make([]uint16, maxInt(size.CDEF.Input, 0)),
		CDEFUnitDst:       make([]uint16, maxInt(size.CDEF.UnitDst, 0)),

		SuperResOutputFrame: make([]byte, maxInt(size.SuperRes.OutputFrame, 0)),

		RestorationData:   make([]uint16, maxInt(size.Restoration.Samples.DataLen, 0)),
		RestorationDst:    make([]uint16, maxInt(size.Restoration.Samples.DstLen, 0)),
		RestorationWiener: make([]uint16, maxInt(size.Restoration.Apply.Unit.Wiener, 0)),
		RestorationSGR:    make([]int32, maxInt(size.Restoration.Apply.Unit.SGRProj, 0)),
		RestorationAbove:  make([]uint16, maxInt(size.Restoration.Apply.Boundary.Above, 0)),
		RestorationBelow:  make([]uint16, maxInt(size.Restoration.Apply.Boundary.Below, 0)),

		FilmGrainOutputFrame: make([]byte, maxInt(size.FilmGrain.OutputFrame, 0)),
		FilmGrainLumaGrain:   make([]int16, maxInt(size.FilmGrain.LumaGrain, 0)),
		FilmGrainLumaSamples: make([]uint16, maxInt(size.FilmGrain.LumaSamples, 0)),
	}
	for plane := 0; plane < 3; plane++ {
		scratch.CDEFSamples[plane] = make([]uint16, maxInt(size.CDEF.Samples[plane], 0))
		scratch.CDEFDst[plane] = make([]uint16, maxInt(size.CDEF.Dst[plane], 0))
		scratch.SuperResCoded[plane] = make([]uint16, maxInt(size.SuperRes.CodedSamples[plane], 0))
		scratch.SuperResOutput[plane] = make([]uint16, maxInt(size.SuperRes.OutputSamples[plane], 0))
	}
	for plane := 0; plane < 2; plane++ {
		scratch.FilmGrainChromaGrain[plane] = make([]int16, maxInt(size.FilmGrain.ChromaGrain[plane], 0))
		scratch.FilmGrainChromaSamples[plane] = make([]uint16, maxInt(size.FilmGrain.ChromaSamples[plane], 0))
	}
	return scratch
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
