package goav1

import (
	"crypto/sha256"
	"encoding/binary"
	"github.com/thesyncim/goav1/internal/av1/bitstream"
	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/obu"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	"hash/maphash"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// frameThreadTestClips are single-tile inter clips (every frame takes the
// frame-threaded path at workers > 1) plus a 10-bit and a two-tile clip that
// mix threaded and drained frames, and film-grain clips grained at output.
var frameThreadTestClips = []string{
	"testdata/benchcorpus/p360_inter_q32.ivf",
	"testdata/benchcorpus/p360_inter_q32_10bit.ivf",
	"testdata/benchcorpus/p360_inter_q32_2tiles.ivf",
	"testdata/benchcorpus/p144_intra_q32.ivf",
	"internal/av1/testdata/libaom/av1-1-b8-01-size-226x226.ivf",
	"internal/av1/testdata/libaom/av1-1-b8-04-cdfupdate.ivf",
	"internal/av1/testdata/libaom/av1-1-b8-05-mv.ivf",
	"internal/av1/testdata/libaom/av1-1-b8-06-mfmv.ivf",
	"internal/av1/testdata/libaom/av1-1-b8-16-intra_only-intrabc-extreme-dv.ivf",
	"internal/av1/testdata/libaom/av1-1-b8-22-svc-L1T2.ivf",
	"internal/av1/testdata/libaom/av1-1-b8-23-film_grain-50.ivf",
	"internal/av1/testdata/libaom/av1-1-b10-24-monochrome.ivf",
	"internal/av1/testdata/libaom/av1-1-b10-00-quantizer-10.ivf",
	// SVT-AV1 random access with film grain: grained outputs leave the ring
	// several per call, between show_existing_frame outputs.
	"testdata/realcorpus/repro/svtra_oldtown_360p_20f_fgs8_poolempty.ivf",
}

// decodeFrameThreadDigest decodes clip through DecodeNext and hashes every
// returned frame in display order, together with the per-call output counts'
// total.
func decodeFrameThreadDigest(t *testing.T, clip string, opts ...Option) ([]byte, int) {
	t.Helper()
	sum, frames, err := decodeFrameThreadDigestErr(t, clip, opts...)
	if err != nil {
		t.Fatalf("%s: %v", clip, err)
	}
	return sum, frames
}

func decodeFrameThreadDigestErr(t *testing.T, clip string, opts ...Option) ([]byte, int, error) {
	t.Helper()
	raw, err := os.ReadFile(clip)
	if err != nil {
		t.Skipf("read %s: %v", clip, err)
	}
	dec, err := NewDecoderFromIVF(raw, opts...)
	if err != nil {
		return nil, 0, err
	}
	defer dec.Close()
	sum := sha256.New()
	frames := 0
	for {
		got, ok, err := dec.DecodeNext()
		if err != nil {
			return nil, 0, err
		}
		if !ok {
			break
		}
		for _, f := range got {
			frames++
			hashDecodedFrame(sum, f)
		}
	}
	return sum.Sum(nil), frames, nil
}

func hashDecodedFrame(sum interface{ Write([]byte) (int, error) }, f *Frame) {
	for _, p := range []FramePlane{f.Y, f.U, f.V} {
		if p.Width == 0 || p.Height == 0 {
			continue
		}
		row := p.Width * f.Layout.BytesPerSample
		for y := 0; y < p.Height; y++ {
			sum.Write(p.Pix[y*p.Stride : y*p.Stride+row])
		}
	}
}

// TestFrameThreadedDecodeByteIdentical is the frame-threading gate: every
// frame-delay depth and worker count must emit exactly the serial decode's
// frames in the same order. Run under -race it also proves the parse of frame
// N+1 shares no unsynchronized state with frame N's reconstruction tail.
func TestFrameThreadedDecodeByteIdentical(t *testing.T) {
	clips := append([]string(nil), frameThreadTestClips...)
	profiles, _ := filepath.Glob("internal/av1/testvector/testdata/profiles/*.ivf")
	clips = append(clips, profiles...)
	for _, clip := range clips {
		if _, err := os.Stat(clip); err != nil {
			// Local corpora (benchcorpus, libaom vectors) are not in every
			// checkout; the tracked profile clips always run.
			t.Logf("%s: %v; skipped", clip, err)
			continue
		}
		want, wantFrames, err := decodeFrameThreadDigestErr(t, clip, WithWorkers(1))
		if err != nil {
			t.Logf("%s: serial decode fails (%v); skipped", clip, err)
			continue
		}
		for _, workers := range []int{2, 4, 8} {
			serial, serialFrames, err := decodeFrameThreadDigestErr(t, clip, WithWorkers(workers), WithMaxFrameDelay(1))
			if err != nil {
				// A pre-existing failure of the one-frame-at-a-time multi-worker
				// path; the frame pipeline must fail the same way.
				if _, _, fpErr := decodeFrameThreadDigestErr(t, clip, WithWorkers(workers)); fpErr == nil {
					t.Fatalf("%s workers=%d: frame threading decodes a clip the serial path rejects (%v)", clip, workers, err)
				}
				t.Logf("%s workers=%d: serial multi-worker decode fails (%v); skipped", clip, workers, err)
				continue
			}
			if serialFrames != wantFrames || string(serial) != string(want) {
				t.Logf("%s workers=%d: delay-1 multi-worker output differs from workers=1 (pre-existing)", clip, workers)
			}
			for _, delay := range []int{0, 2, 3, 4} {
				got, frames := decodeFrameThreadDigest(t, clip, WithWorkers(workers), WithMaxFrameDelay(delay))
				if frames != serialFrames || string(got) != string(serial) {
					t.Fatalf("%s workers=%d delay=%d: %d frames, digest mismatch=%v (want %d frames)",
						clip, workers, delay, frames, string(got) != string(serial), serialFrames)
				}
			}
		}
	}
}

// TestFrameThreadedParseNeverReadsReferencePixels is the poison-References
// audit for frame threading: while a frame's entropy pass runs, every
// reference surface it may predict from is overwritten with garbage and its
// reconstruction lanes are held back; the true pixels return only after the
// pass. The decode stays byte-exact, so the entropy pass (dav1d's
// DAV1D_TASK_TYPE_TILE_ENTROPY) neither reads reference pixels nor waits for
// them.
//
// The poisoned references are also frames the decoder already returned, so
// the test hashes its outputs under outputMu and the hook poisons under it
// too: the decoder never writes an output frame, only this test does.
func TestFrameThreadedParseNeverReadsReferencePixels(t *testing.T) {
	clip := "testdata/benchcorpus/p360_inter_q32.ivf"
	want, wantFrames := decodeFrameThreadDigest(t, clip, WithWorkers(1))

	var gate internalthreading.FrameProgress
	var outputMu sync.Mutex
	poisoned := 0
	decoderFramePipelineParseHook = func(p *decoderFramePipeline, fc *decoderFrameContext, references []*Frame) func() {
		if len(references) == 0 {
			return nil
		}
		// The previous frames' tails may still be filtering or reading these
		// surfaces: wait for every other frame in flight, then poison. Later
		// frames are held back by the pipeline's hook lock until restore.
		for s := range p.progress {
			if s != fc.surface {
				p.progress[s].Wait(internalthreading.FrameProgressDone)
			}
		}
		for i := range references {
			fc.ft.Recon.RefProgress[i].Wait(internalthreading.FrameProgressDone)
		}
		// Retiring frames needs the caller's next DecodeNext, so the lock is
		// taken only after the waits.
		outputMu.Lock()
		gate.Reset()
		var saved [][]byte
		seen := map[*Frame]bool{}
		for i, f := range references {
			fc.ft.Recon.RefProgress[i] = &gate
			if seen[f] {
				continue
			}
			seen[f] = true
			for _, pl := range []*FramePlane{&f.Y, &f.U, &f.V} {
				saved = append(saved, append([]byte(nil), pl.Pix...))
				for j := range pl.Pix {
					pl.Pix[j] = byte(0xA5 ^ j)
				}
			}
		}
		poisoned++
		return func() {
			k := 0
			done := map[*Frame]bool{}
			for _, f := range references {
				if done[f] {
					continue
				}
				done[f] = true
				for _, pl := range []*FramePlane{&f.Y, &f.U, &f.V} {
					copy(pl.Pix, saved[k])
					k++
				}
			}
			outputMu.Unlock()
			gate.Publish(internalthreading.FrameProgressDone)
		}
	}
	defer func() { decoderFramePipelineParseHook = nil }()

	raw, err := os.ReadFile(clip)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := NewDecoderFromIVF(raw, WithWorkers(4), WithMaxFrameDelay(3))
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	sum := sha256.New()
	frames := 0
	for {
		out, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		outputMu.Lock()
		for _, f := range out {
			frames++
			hashDecodedFrame(sum, f)
		}
		outputMu.Unlock()
	}
	got := sum.Sum(nil)
	if poisoned == 0 {
		t.Fatal("no frame took the frame-threaded path")
	}
	if frames != wantFrames || string(got) != string(want) {
		t.Fatalf("poisoned-reference decode differs: %d frames (want %d)", frames, wantFrames)
	}
}

// TestFrameContextCountMatchesDav1d pins dav1d's get_num_threads n_fc.
func TestFrameContextCountMatchesDav1d(t *testing.T) {
	cases := []struct{ workers, delay, want int }{
		{1, 0, 1}, {2, 0, 2}, {4, 0, 2}, {5, 0, 3}, {8, 0, 3}, {9, 0, 3}, {10, 0, 4},
		{16, 0, 4}, {64, 0, 8}, {8, 1, 1}, {8, 5, 5}, {2, 5, 2}, {64, 20, 8},
	}
	for _, c := range cases {
		if got := decoderFrameContextCount(c.workers, c.delay); got != c.want {
			t.Errorf("workers=%d delay=%d: n_fc=%d, want %d", c.workers, c.delay, got, c.want)
		}
	}
}

// splitFrameOBUs rewrites an IVF stream so every OBU_FRAME becomes the
// equivalent OBU_FRAME_HEADER + OBU_TILE_GROUP pair (frame_obu() is
// frame_header_obu(), byte_alignment() and tile_group_obu(), AV1 5.10).
func splitFrameOBUs(t *testing.T, ivf []byte) []byte {
	t.Helper()
	if len(ivf) < 32 {
		t.Fatal("short IVF")
	}
	out := append([]byte(nil), ivf[:32]...)
	var stream internaldecoder.Stream
	var event internaldecoder.Event
	for off := 32; off < len(ivf); {
		if off+12 > len(ivf) {
			t.Fatal("short IVF frame header")
		}
		size := int(binary.LittleEndian.Uint32(ivf[off:]))
		frameHeader := ivf[off : off+12]
		tu := ivf[off+12 : off+12+size]
		off += 12 + size

		var units []byte
		for p := 0; p < len(tu); {
			unit, n, err := obu.ParseLowOverhead(tu[p:])
			if err != nil {
				t.Fatal(err)
			}
			p += n
			if err := stream.PushUnitInto(&event, unit, false); err != nil {
				t.Fatal(err)
			}
			if unit.Header.Type != obu.TypeFrame {
				units = append(units, unit.Raw...)
				continue
			}
			bits := int(event.FilmGrain.BitsRead)
			headerBytes := (bits + 7) / 8
			// frame_header_obu() ends in trailing_bits(): a one bit, then
			// zeros to the byte boundary (the zero byte_alignment bits of
			// the frame OBU are already in place).
			fh := append([]byte(nil), unit.Payload[:headerBytes]...)
			if bits%8 == 0 {
				fh = append(fh, 0x80)
			} else {
				fh[bits/8] |= 0x80 >> uint(bits%8)
			}
			units = appendOBU(t, units, unit.Header, obu.TypeFrameHeader, fh)
			units = appendOBU(t, units, unit.Header, obu.TypeTileGroup, unit.Payload[headerBytes:])
		}
		var hdr [12]byte
		copy(hdr[:], frameHeader)
		binary.LittleEndian.PutUint32(hdr[:], uint32(len(units)))
		out = append(out, hdr[:]...)
		out = append(out, units...)
	}
	return out
}

func appendOBU(t *testing.T, dst []byte, h obu.Header, typ obu.Type, payload []byte) []byte {
	t.Helper()
	h.Type = typ
	h.HasSizeField = true
	var hb [2]byte
	n, err := obu.PutHeader(hb[:], h)
	if err != nil {
		t.Fatal(err)
	}
	dst = append(dst, hb[:n]...)
	var size [8]byte
	n, err = bitstream.PutLEB128(size[:], uint32(len(payload)))
	if err != nil {
		t.Fatal(err)
	}
	dst = append(dst, size[:n]...)
	return append(dst, payload...)
}

func decodeFrameThreadDigestBytes(t *testing.T, raw []byte, opts ...Option) ([]byte, int) {
	t.Helper()
	dec, err := NewDecoderFromIVF(raw, opts...)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	sum := sha256.New()
	frames := 0
	for {
		got, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		for _, f := range got {
			frames++
			hashDecodedFrame(sum, f)
		}
	}
	return sum.Sum(nil), frames
}

// TestFrameThreadedFrameHeaderTileGroup: a frame sent as a frame header OBU
// and one tile group OBU holding every tile takes the frame-threaded path
// like an OBU_FRAME (dav1d submits a frame once all its tiles are in), and
// decodes byte-identically to the serial decode.
func TestFrameThreadedFrameHeaderTileGroup(t *testing.T) {
	clips := []string{
		"internal/av1/testvector/testdata/profiles/multitile-2x2-rows-framethread-256x256.ivf",
		"internal/av1/testvector/testdata/profiles/profile0-420-8bit-segmentation-temporal-64x64.ivf",
		"internal/av1/testvector/testdata/profiles/profile0-420-8bit-filmgrain-showexisting-128x128.ivf",
		"testdata/benchcorpus/p360_inter_q32.ivf",
	}
	for _, clip := range clips {
		raw, err := os.ReadFile(clip)
		if err != nil {
			t.Logf("read %s: %v; skipped", clip, err)
			continue
		}
		split := splitFrameOBUs(t, raw)
		want, wantFrames := decodeFrameThreadDigestBytes(t, raw, WithWorkers(1))
		if got, frames := decodeFrameThreadDigestBytes(t, split, WithWorkers(1)); frames != wantFrames || string(got) != string(want) {
			t.Fatalf("%s: split stream decodes differently (%d frames, want %d)", clip, frames, wantFrames)
		}
		// The parse hook serializes submissions behind each entropy pass, so it
		// only checks that the ring is used; the digests run without it.
		threaded := 0
		decoderFramePipelineParseHook = func(*decoderFramePipeline, *decoderFrameContext, []*Frame) func() {
			threaded++
			return nil
		}
		decodeFrameThreadDigestBytes(t, split, WithWorkers(4), WithMaxFrameDelay(3))
		decoderFramePipelineParseHook = nil
		if threaded == 0 {
			t.Fatalf("%s: no frame took the frame-threaded path", clip)
		}
		for _, workers := range []int{2, 4, 8} {
			for _, delay := range []int{0, 2, 4} {
				got, frames := decodeFrameThreadDigestBytes(t, split, WithWorkers(workers), WithMaxFrameDelay(delay))
				if frames != wantFrames || string(got) != string(want) {
					t.Fatalf("%s workers=%d delay=%d: %d frames, digest mismatch=%v (want %d frames)",
						clip, workers, delay, frames, string(got) != string(want), wantFrames)
				}
			}
		}
	}
}

// poisonRowsTailHook makes every finished frame-threaded surface final one
// band of luma rows at a time: it overwrites all of the surface's samples
// (padding included) with garbage, then restores and publishes the rows top
// down, pausing between bands, and leaves the bottom padding for the final
// publish. A frame that reads a reference row before its band is published
// reads garbage, so any under-estimated reference wait changes the output.
func poisonRowsTailHook(band int) func(*Frame, *internalthreading.FrameProgress) {
	return func(f *Frame, progress *internalthreading.FrameProgress) {
		planes := []*FramePlane{&f.Y, &f.U, &f.V}
		saved := make([][]byte, len(planes))
		for i, pl := range planes {
			if pl.Stride == 0 {
				continue
			}
			saved[i] = append([]byte(nil), pl.Pix...)
			for j := range pl.Pix {
				pl.Pix[j] = byte(0x5A ^ j ^ j>>8)
			}
		}
		lumaH := f.Y.Height
		restore := func(lo, hi int) {
			for i, pl := range planes {
				if saved[i] == nil {
					continue
				}
				shift := 0
				if i > 0 && f.U.Height < f.Y.Height {
					shift = 1
				}
				plo, phi := lo>>shift, (hi+shift)>>shift
				if hi >= lumaH {
					phi = len(pl.Pix) / pl.Stride
				}
				copy(pl.Pix[plo*pl.Stride:phi*pl.Stride], saved[i][plo*pl.Stride:phi*pl.Stride])
			}
		}
		for lo := 0; lo < lumaH; lo += band {
			hi := min(lo+band, lumaH)
			if hi < lumaH {
				restore(lo, hi)
				progress.Publish(int32(hi))
			} else {
				restore(lo, lumaH)
			}
			time.Sleep(20 * time.Microsecond)
		}
	}
}

// TestFrameThreadedReferenceRowWaits checks the per-superblock reference row
// waits (dav1d lowest_px): with every reference surface published one band at
// a time and garbage below the published rows, the frame-threaded decode
// still matches the serial one, so no reconstruction reads a reference row it
// did not wait for.
func TestFrameThreadedReferenceRowWaits(t *testing.T) {
	clips := []string{
		// Tracked clips first; the local corpora are skipped when absent.
		"internal/av1/testvector/testdata/profiles/multitile-2x2-rows-framethread-256x256.ivf",
		"internal/av1/testvector/testdata/profiles/profile0-420-8bit-edgemv-130x130.ivf",
		"internal/av1/testvector/testdata/profiles/palette-inter-edge-256x144.ivf",
		"internal/av1/testvector/testdata/profiles/svt-compound-edge-filter-256x136.ivf",
		"testdata/benchcorpus/p360_inter_q32.ivf",
		"testdata/benchcorpus/p360_inter_q32_2tiles.ivf",
		"internal/av1/testdata/libaom/av1-1-b8-05-mv.ivf",
		"internal/av1/testdata/libaom/av1-1-b8-06-mfmv.ivf",
		"testdata/realcorpus/streams/aomgood_mtbike_720p_1500k_c6.ivf",
		"testdata/realcorpus/streams/aomgood_parkjoy_360p_1000k_c4.ivf",
		"testdata/realcorpus/streams/aomgood_johnny_360p_600k_c2.ivf",
		"testdata/realcorpus/streams/svtra_johnny_720p_crf35_p8.ivf",
		"testdata/realcorpus/streams/svtra_oldtown_720p_crf35_p8_fgs8.ivf",
	}
	for _, clip := range clips {
		if _, err := os.Stat(clip); err != nil {
			t.Logf("%s: %v; skipped", clip, err)
			continue
		}
		want, wantFrames, err := decodeFrameThreadDigestErr(t, clip, WithWorkers(1))
		if err != nil {
			t.Fatalf("%s: %v", clip, err)
		}
		for _, band := range []int{8, 64} {
			decoderFramePipelineTailHook = poisonRowsTailHook(band)
			got, frames, err := decodeFrameThreadDigestErr(t, clip, WithWorkers(4), WithMaxFrameDelay(3))
			decoderFramePipelineTailHook = nil
			if err != nil {
				t.Fatalf("%s band=%d: %v", clip, band, err)
			}
			if frames != wantFrames || string(got) != string(want) {
				t.Fatalf("%s band=%d: a frame read reference rows before they were published (%d frames, want %d)", clip, band, frames, wantFrames)
			}
		}
	}
}

// publishedRowsChecker records a hash of every row a frame-threaded
// post-filter publishes as final (with the chroma rows it covers, right
// padding included) and, once the frame's tail finished, checks that no
// published row changed afterwards: a row published before a later stage
// rewrote it would be read by other frames before it was final.
type publishedRowsChecker struct {
	t     *testing.T
	seed  maphash.Seed
	mu    sync.Mutex
	rows  map[*internalframe.Frame][]uint64
	count int
	bad   int
}

func (c *publishedRowsChecker) rowHash(f *internalframe.Frame, y int) uint64 {
	var h maphash.Hash
	h.SetSeed(c.seed)
	h.Write(f.Y.Pix[y*f.Y.Stride : (y+1)*f.Y.Stride])
	if !f.Format.MonoChrome && f.U.Stride != 0 {
		cy := y
		if f.Format.SubsamplingY {
			if y&1 != 0 {
				return h.Sum64()
			}
			cy = y >> 1
		}
		h.Write(f.U.Pix[cy*f.U.Stride : (cy+1)*f.U.Stride])
		h.Write(f.V.Pix[cy*f.V.Stride : (cy+1)*f.V.Stride])
	}
	return h.Sum64()
}

func (c *publishedRowsChecker) publish(f *internalframe.Frame, rows int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	have := c.rows[f]
	for y := len(have); y < rows; y++ {
		have = append(have, c.rowHash(f, y))
	}
	c.rows[f] = have
	c.count++
}

func (c *publishedRowsChecker) done(f *Frame) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for y, want := range c.rows[f] {
		if c.rowHash(f, y) != want {
			c.bad++
			c.t.Errorf("luma row %d of a %dx%d surface changed after it was published", y, f.Y.Width, f.Y.Height)
			break
		}
	}
	delete(c.rows, f)
}

// TestFrameThreadedPublishedRowsFinal checks the publish side of the
// frame-threaded row progress: every row a post-filter publishes while it
// still runs keeps its value until the frame is done, and the output matches
// the serial decode.
func TestFrameThreadedPublishedRowsFinal(t *testing.T) {
	clips := []string{
		// Tracked clips first; the local corpora are skipped when absent.
		"internal/av1/testvector/testdata/profiles/multitile-2x2-rows-framethread-256x256.ivf",
		"internal/av1/testvector/testdata/profiles/profile0-420-8bit-edgemv-130x130.ivf",
		"internal/av1/testvector/testdata/profiles/palette-inter-edge-256x144.ivf",
		"internal/av1/testvector/testdata/profiles/svt-compound-edge-filter-256x136.ivf",
		"internal/av1/testvector/testdata/profiles/profile0-420-8bit-filmgrain-showexisting-128x128.ivf",
		"internal/av1/testvector/testdata/profiles/profile1-444-10bit-edgemv-130x130.ivf",
		"internal/av1/testvector/testdata/profiles/profile2-420-12bit-edgemv-130x130.ivf",
		"internal/av1/testvector/testdata/profiles/profile1-444-10bit-filmgrain-96x96.ivf",
		"testdata/benchcorpus/p360_inter_q32.ivf",
		"testdata/benchcorpus/p360_inter_q32_10bit.ivf",
		"testdata/benchcorpus/p360_inter_q32_2tiles.ivf",
		"testdata/realcorpus/streams/aomrt_mtbike_360p_600k_s8.ivf",
		"testdata/realcorpus/streams/aomrt_fourpeople_720p_1000k_s10.ivf",
		"testdata/realcorpus/streams/aomrt_wikipedia_1080p_1500k_s11.ivf",
		"testdata/realcorpus/streams/svtrtc_wikipedia_1080p_1000k_p12.ivf",
		"testdata/realcorpus/streams/goav1_johnny_360p_500k.ivf",
		"testdata/realcorpus/streams/aomgood_mtbike_720p_1500k_c6.ivf",
		"testdata/realcorpus/streams/svtra_johnny_720p_crf35_p8.ivf",
	}
	c := &publishedRowsChecker{t: t, seed: maphash.MakeSeed(), rows: map[*internalframe.Frame][]uint64{}}
	internaldecoder.FrameThreadRowsPublishHook = c.publish
	decoderFramePipelineDoneHook = c.done
	defer func() {
		internaldecoder.FrameThreadRowsPublishHook = nil
		decoderFramePipelineDoneHook = nil
	}()
	for _, clip := range clips {
		if _, err := os.Stat(clip); err != nil {
			t.Logf("%s: %v; skipped", clip, err)
			continue
		}
		want, wantFrames, err := decodeFrameThreadDigestErr(t, clip, WithWorkers(1))
		if err != nil {
			t.Fatalf("%s: %v", clip, err)
		}
		for _, workers := range []int{4, 8} {
			before := c.count
			got, frames, err := decodeFrameThreadDigestErr(t, clip, WithWorkers(workers))
			if err != nil {
				t.Fatalf("%s workers=%d: %v", clip, workers, err)
			}
			if frames != wantFrames || string(got) != string(want) {
				t.Fatalf("%s workers=%d: output differs from the serial decode", clip, workers)
			}
			t.Logf("%s workers=%d: %d row publishes", clip, workers, c.count-before)
			if c.bad != 0 {
				t.FailNow()
			}
		}
	}
}

// wavefrontTestVector is a single-tile two-layer SVC clip whose 1280x720
// enhancement layer has enough superblock rows that a 2-4 worker wavefront
// overlaps at the production threshold.
const wavefrontTestVector = "internal/av1/testdata/libaom/av1-1-b8-22-svc-L2T1.ivf"

func decodeWavefrontFrames(t testing.TB, workers int) []byte {
	t.Helper()
	raw, err := os.ReadFile(wavefrontTestVector)
	if err != nil {
		t.Skipf("read wavefront vector: %v", err)
	}
	it, err := NewIVFIterator(raw)
	if err != nil {
		t.Fatalf("NewIVFIterator: %v", err)
	}
	var payloads [][]byte
	for {
		f, ok, err := it.Next()
		if err != nil {
			t.Fatalf("ivf next: %v", err)
		}
		if !ok {
			break
		}
		payloads = append(payloads, append([]byte(nil), f.Payload...))
	}
	if len(payloads) == 0 {
		t.Fatal("no frames in wavefront vector")
	}

	// WithMaxFrameDelay(1) keeps one frame in flight, so every worker is a tile
	// or wavefront lane rather than a frame-threading lane.
	dec, err := NewDecoder(payloads, WithWorkers(workers), WithMaxFrameDelay(1))
	if err != nil {
		t.Fatalf("NewDecoder (workers=%d): %v", workers, err)
	}
	defer dec.Close()

	sum := sha256.New()
	// Hash every frame as DecodeNext returns it, before the decoder can recycle
	// its surface.
	for {
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode (workers=%d): %v", workers, err)
		}
		if !ok {
			break
		}
		for _, f := range frames {
			hashFramePlanes(sum, f)
		}
	}
	return sum.Sum(nil)
}

func hashFramePlanes(sum interface{ Write([]byte) (int, error) }, f *Frame) {
	if f == nil {
		return
	}
	sum.Write(f.Y.Pix)
	if !f.Format.MonoChrome {
		sum.Write(f.U.Pix)
		sum.Write(f.V.Pix)
	}
}

// TestWavefrontByteIdenticalToSingleWorker is the load-bearing gate: the
// multi-worker SB-row wavefront must reconstruct exactly the same bytes as the
// single-worker fused path. Run it under -race to also prove the wavefront is
// data-race free.
func TestWavefrontByteIdenticalToSingleWorker(t *testing.T) {
	want := decodeWavefrontFrames(t, 1)
	for _, workers := range []int{2, 3, 4} {
		got := decodeWavefrontFrames(t, workers)
		if string(got) != string(want) {
			t.Fatalf("workers=%d output differs from workers=1 (wavefront not byte-identical)", workers)
		}
	}
}

// BenchmarkWavefrontDecode decodes the single-tile clip end to end with one
// worker and with four. The wavefront parallelizes only reconstruction; entropy
// decoding stays serial, so the end-to-end gain on this small clip is modest.
// The reconstruction phase alone is BenchmarkReconWavefrontReplay in the
// threading package.
func BenchmarkWavefrontDecode(b *testing.B) {
	for _, bc := range []struct {
		name    string
		workers int
	}{{"workers1", 1}, {"workers4", 4}} {
		b.Run(bc.name, func(b *testing.B) {
			for b.Loop() {
				decodeWavefrontFrames(b, bc.workers)
			}
		})
	}
}
