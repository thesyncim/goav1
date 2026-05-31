// Package conformance guards the PUBLIC goav1 decode path against libaom's
// per-frame MD5 goldens. It lives in its own package and test binary so it
// shares no process state with the internal/av1/testvector oracle suite, and
// it deliberately drives only the exported goav1 API (the residual stream
// runner plus the reusable supported post-filter runner) -- the same path a
// production integration and cmd/aom-go-dec take. If this test passes, the
// public stream-runner path reconstructs and post-filters byte-for-byte like
// libaom for the covered clips.
package conformance

import (
	"bufio"
	"bytes"
	"crypto/md5"
	"encoding/hex"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

type publicClip struct {
	file        string
	frameMD5Hex []string
}

// These vendored profile-conformance clips and their aomdec per-frame MD5
// goldens are shared with internal/av1/testvector/profiles. They cover
// profile-1 4:4:4 8/10-bit, profile-1 screen-content palette, profile-2
// 4:2:2 8-bit, and profile-2 4:2:0 12-bit -- including inter-coded non-4:2:0
// clips that libaom's published vector suite does not ship.
var publicClips = []publicClip{
	{
		file: "profile1-444-8bit-64x64.ivf",
		frameMD5Hex: []string{
			"00211cdc8f799c808849c955a318a0f5",
			"397ff01920ff514bc611ab49d76371c1",
			"f8fbfb25a42da47a7adb71510de9b178",
		},
	},
	{
		file: "profile1-444-8bit-palette-64x64.ivf",
		frameMD5Hex: []string{
			"59e96ccf6f2faf403ae8f8b3214a202f",
			"9a9cfad7d6aead1fb110309d2edc0268",
			"6d7426c5bae2c5cca7c98b584e8057a6",
		},
	},
	{
		file: "profile1-444-10bit-64x64.ivf",
		frameMD5Hex: []string{
			"96322d4430f0d27243c17e2128cfd625",
			"571c0415f63a1f76e3796360aee3828a",
			"2f6e92f93a95cb4725b9c2f9484ed42e",
		},
	},
	{
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
	},
	{
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
	},
	{
		file: "profile2-422-8bit-64x64.ivf",
		frameMD5Hex: []string{
			"dabd492413632a810adeaf4e5d0c6d97",
			"eb6cf8d1d4d644686cf03c513acd5978",
			"84983e98afeef6692448e98fc5980431",
		},
	},
	{
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
	},
	{
		file: "profile2-420-12bit-64x64.ivf",
		frameMD5Hex: []string{
			"e714741e4ad4fce5a4469c79705f132c",
			"447103c7d7358e4cbb6f5b98ce4e1be1",
			"dcd86cbbf80f81d9699eaf6c6e879e72",
		},
	},
}

var callerPostFilterClips = []publicClip{
	{
		file: "profile1-444-8bit-superres-160x128.ivf",
		frameMD5Hex: []string{
			"315663b1b307e46d4d4719ad759b39c2",
			"b1c26bbbf277ac82c7fbdc0807974a1f",
			"bd6aad53bf088c7ff5f54d93c59ff649",
			"775085d7d08c8608b0acd2defa0fdc58",
		},
	},
	{
		file: "profile1-444-10bit-superres-160x128.ivf",
		frameMD5Hex: []string{
			"c556c65e1ceceb72778a36a5351b1d96",
			"6e66c5c97438754259295a95dde348a4",
			"cbf0f9d0584eaaf09538091edfb8f885",
			"f6f4c4213052d92f0ae700e793cad629",
		},
	},
	{
		file: "superres-420-8bit-160x128.ivf",
		frameMD5Hex: []string{
			"024e5d1f55340eee39893567e8000554",
			"80c239661f2cd1afe5fce5038ceac76f",
			"93dc5cde59cb99bd524264b15c70a700",
			"87887afdc06ba5f0e88b2acbe1ce0ce5",
		},
	},
	{
		file: "superres-restoration-420-8bit-160x128.ivf",
		frameMD5Hex: []string{
			"93533110e2bd60fed6f15b5e65178bc6",
			"bf83a513a868584aec7f9f05c5187d70",
			"4e7dc4a0d7ab9cebdbdfdf3ef73b0d1c",
			"6af9472b0679093facbefa85e973a7f9",
		},
	},
}

// TestPublicPathProfileClips decodes each vendored profile clip through the
// public stream runner and asserts every visible frame's MD5 matches the
// libaom golden.
func TestPublicPathProfileClips(t *testing.T) {
	for _, clip := range publicClips {
		t.Run(clip.file, func(t *testing.T) {
			got := decodeClipPublic(t, clip.file)
			if len(got) != len(clip.frameMD5Hex) {
				t.Fatalf("decoded %d visible frames, want %d", len(got), len(clip.frameMD5Hex))
			}
			for i, want := range clip.frameMD5Hex {
				if g := hex.EncodeToString(got[i][:]); g != want {
					t.Fatalf("frame %d md5 got=%s want=%s (libaom golden)", i, g, want)
				}
			}
		})
	}
}

// TestPublicPathCallerPostFilterProfileClips drives all-keyframe super-res
// clips through the exported caller-owned full postfilter runner. This covers
// display output for super-res and post-super-res restoration without requiring
// publishable upscaled references.
func TestPublicPathCallerPostFilterProfileClips(t *testing.T) {
	for _, clip := range callerPostFilterClips {
		t.Run(clip.file, func(t *testing.T) {
			var postFilter av1.DecoderFrameWorkReusableCallerPostFilterRunner
			got := decodeClipPublicDataWithRunner(t, readPublicClip(t, clip.file), &postFilter)
			if len(got) != len(clip.frameMD5Hex) {
				t.Fatalf("decoded %d visible frames, want %d", len(got), len(clip.frameMD5Hex))
			}
			for i, want := range clip.frameMD5Hex {
				if g := hex.EncodeToString(got[i][:]); g != want {
					t.Fatalf("frame %d md5 got=%s want=%s (libaom golden)", i, g, want)
				}
			}
		})
	}
}

// TestPublicPathLibaomQuantizer00 decodes the vendored libaom 4:2:0 8-bit
// quantizer-00 vector through the public stream runner and asserts each visible
// frame matches the official libaom per-frame MD5. This is the regression that
// previously produced a wrong per-frame MD5 on the public path.
func TestPublicPathLibaomQuantizer00(t *testing.T) {
	const ivfName = "av1-1-b8-00-quantizer-00.ivf"
	// The libaom vectors under internal/av1/testdata/libaom are downloaded,
	// not committed; skip when they are absent (the vendored profile clips in
	// TestPublicPathProfileClips already guard the public path unconditionally).
	if !libaomClipPresent(ivfName) {
		t.Skipf("vendored libaom vector %s not present", ivfName)
	}
	want := readLibaomFrameMD5(t, ivfName+".md5")
	got := decodeLibaomClipPublic(t, ivfName)
	if len(got) != len(want) {
		t.Fatalf("decoded %d visible frames, want %d", len(got), len(want))
	}
	for i := range want {
		if g := hex.EncodeToString(got[i][:]); g != want[i] {
			t.Fatalf("frame %d md5 got=%s want=%s (libaom golden)", i, g, want[i])
		}
	}
}

// readLibaomFrameMD5 parses a libaom ".md5" golden (one "hexdigest  filename"
// line per visible frame) and returns the per-frame hex digests in order.
func readLibaomFrameMD5(t *testing.T, name string) []string {
	t.Helper()
	data, err := os.ReadFile(libaomClipPath(name))
	if err != nil {
		t.Fatalf("read md5 %s: %v", name, err)
	}
	var digests []string
	sc := bufio.NewScanner(bytes.NewReader(data))
	for sc.Scan() {
		fields := bytes.Fields(sc.Bytes())
		if len(fields) == 0 {
			continue
		}
		digests = append(digests, string(fields[0]))
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan md5: %v", err)
	}
	return digests
}

func decodeLibaomClipPublic(t *testing.T, file string) [][16]byte {
	t.Helper()
	return decodeClipPublicData(t, readLibaomClip(t, file))
}

// decodeClipPublic runs one IVF clip through the exported goav1 stream-runner
// path with the reusable supported post-filter runner and returns the MD5 of
// every visible output frame in display order.
func decodeClipPublic(t *testing.T, file string) [][16]byte {
	t.Helper()
	return decodeClipPublicData(t, readPublicClip(t, file))
}

func decodeClipPublicData(t *testing.T, ivfData []byte) [][16]byte {
	t.Helper()
	var postFilter av1.DecoderFrameWorkReusableSupportedPostFilterRunner
	return decodeClipPublicDataWithRunner(t, ivfData, &postFilter)
}

func decodeClipPublicDataWithRunner(t *testing.T, ivfData []byte, postFilter av1.DecoderFrameWorkPostFilterRunner) [][16]byte {
	t.Helper()

	it, err := av1.NewIVFIterator(ivfData)
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
		// Copy: the iterator may reuse its payload backing.
		payloads = append(payloads, append([]byte(nil), f.Payload...))
	}
	if len(payloads) == 0 {
		t.Fatal("clip produced no frames")
	}

	const workers = 1
	workerPool, err := av1.NewTileWorkerPool(workers)
	if err != nil {
		t.Fatalf("worker pool: %v", err)
	}
	defer workerPool.Close()

	var probeStream av1.DecoderStream
	probeEvents := make([]av1.DecoderEvent, 16*len(payloads)+64)
	probeSpans := make([]av1.TileSpan, av1.MaxTiles)
	probeJobs := make([]av1.TileJob, av1.MaxTiles)
	probeBatches := make([]av1.TileBatch, av1.MaxTiles)
	plan, err := av1.DecoderFrameWorkResidualLowOverheadStreamsPlan(
		probeStream, payloads, workers, probeEvents, probeSpans, probeJobs, probeBatches)
	if err != nil {
		t.Fatalf("stream plan: %v", err)
	}
	if !plan.HasEvent() {
		t.Fatal("stream plan did not identify a bind event")
	}

	// Build the pool format the way a production caller should: from the bound
	// sequence/frame headers via the public helper, so the superblock alignment
	// (64 vs 128) is derived from the stream rather than guessed.
	format, err := av1.FrameCodedFormatFromHeaders(plan.Bind.Sequence, plan.Bind.Event.FrameSize, 64)
	if err != nil {
		t.Fatalf("FrameCodedFormatFromHeaders: %v", err)
	}

	const surfaceCount = av1.RefFrames + 1
	pool := bindPublicFramePool(t, format, surfaceCount)

	var (
		stream   av1.DecoderStream
		refs     av1.DecoderSurfaceReferences
		state    av1.DecoderFrameWorkState
		stats    av1.DecoderFrameWorkTileResidualStats
		sideData av1.DecoderFrameWorkSideData
		batch    av1.DecoderFrameWorkBatchResidualRunner
	)
	refSurface := make([]int, av1.InterRefsPerFrame)
	refFrames := make([]*av1.Frame, av1.InterRefsPerFrame)
	releases := make([]int, av1.RefFrames)

	scratch := newPublicStreamScratch(plan.Size)
	runner, _, err := av1.BindDecoderFrameWorkResidualStreamPlanRunner(plan, &stream,
		av1.DecoderFrameWorkResidualEventRuntime{
			State:             &state,
			Refs:              &refs,
			FramePool:         &pool,
			Align:             64,
			ReferenceSurfaces: refSurface,
			ReferenceFrames:   refFrames,
			Releases:          releases,
			WorkerPool:        workerPool,
			SideData:          &sideData,
			Stats:             &stats,
		}, scratch, &batch)
	if err != nil {
		t.Fatalf("bind runner: %v", err)
	}

	var digests [][16]byte
	for i, payload := range payloads {
		var result av1.DecoderFrameWorkResidualStreamResult
		if err := runner.RunLowOverheadIntoWithPostFilterRunner(&result, payload, postFilter); err != nil {
			t.Fatalf("frame %d run: %v", i, err)
		}
		for _, frame := range result.Run.Outputs {
			if frame == nil {
				continue
			}
			digests = append(digests, frameMD5(t, frame))
		}
	}
	return digests
}

// frameMD5 reproduces the libaom test/md5_helper.h per-frame digest layout:
// visible Y rows, then visible U rows, then visible V rows, each stripped of
// stride padding. This matches internal/av1/testvector.FrameMD5 and the
// aomdec --rawvideo goldens.
func frameMD5(t *testing.T, f *av1.Frame) [16]byte {
	t.Helper()
	h := md5.New()
	bytesPerSample := f.Layout.BytesPerSample
	for _, plane := range []av1.FramePlane{f.Y, f.U, f.V} {
		if plane.Width == 0 || plane.Height == 0 || len(plane.Pix) == 0 {
			continue
		}
		rowBytes := plane.Width * bytesPerSample
		for row := 0; row < plane.Height; row++ {
			off := row * plane.Stride
			if _, err := h.Write(plane.Pix[off : off+rowBytes]); err != nil {
				t.Fatalf("md5 write: %v", err)
			}
		}
	}
	var sum [16]byte
	h.Sum(sum[:0])
	return sum
}

func bindPublicFramePool(t *testing.T, format av1.FrameFormat, count int) av1.FramePool {
	t.Helper()
	_, backingSize, err := av1.FramePoolRequiredSize(format, count)
	if err != nil {
		t.Fatalf("FramePoolRequiredSize: %v", err)
	}
	pool, err := av1.BindFramePool(make([]byte, backingSize), format,
		make([]av1.Frame, count), make([]int, count), make([]bool, count))
	if err != nil {
		t.Fatalf("BindFramePool: %v", err)
	}
	return pool
}

func newPublicStreamScratch(size av1.DecoderFrameWorkResidualStreamScratchSize) av1.DecoderFrameWorkResidualStreamScratch {
	return av1.DecoderFrameWorkResidualStreamScratch{
		Events:    make([]av1.DecoderEvent, size.Events),
		Event:     newPublicEventScratch(size.Event),
		SideData:  newPublicSideDataScratch(size.Event.SideData),
		Outputs:   make([]*av1.Frame, size.Event.Outputs),
		RTPBuffer: make([]byte, size.RTPBuffer),
		RTPSpans:  make([]av1.RTPObuSpan, size.RTPSpans),
	}
}

func newPublicEventScratch(size av1.DecoderFrameWorkResidualEventScratchSize) av1.DecoderFrameWorkResidualEventScratch {
	return av1.DecoderFrameWorkResidualEventScratch{
		Runner:   newPublicBatchRunnerScratch(size.Runner),
		SideData: newPublicSideDataScratch(size.SideData),
		Spans:    make([]av1.TileSpan, size.Plan.SpanCount),
		Jobs:     make([]av1.TileJob, size.Plan.JobCount),
		Batches:  make([]av1.TileBatch, size.Plan.BatchCount),
	}
}

func newPublicBatchRunnerScratch(size av1.DecoderFrameWorkBatchResidualRunnerScratchSize) av1.DecoderFrameWorkBatchResidualRunnerScratch {
	return av1.DecoderFrameWorkBatchResidualRunnerScratch{
		States:                  make([]av1.TileDecodeState, size.Workers),
		Storages:                make([]av1.DecoderFrameWorkTileResidualCDFStorage, size.Workers),
		TileScratch:             make([]av1.DecoderFrameWorkTileResidualScratch, size.Workers),
		RestorationRequests:     make([]av1.DecoderFrameWorkTileRestorationRequest, size.RestorationRequests),
		PredictionScratch:       make([]av1.DecoderFrameWorkPredictionScratch, size.Workers),
		InterPredictionScratch:  make([]av1.DecoderFrameWorkInterPredictionScratch, size.Workers),
		Stats:                   make([]av1.DecoderFrameWorkTileResidualStats, size.Workers),
		Int32Scratch:            make([]int32, size.Int32Scratch),
		ResidualScratch:         make([]int16, size.ResidualScratch),
		LoopContextAboveScratch: make([]av1.TileBlockLoopRootAboveContext, size.LoopContextAbove),
	}
}

func newPublicSideDataScratch(size av1.DecoderFrameWorkSideDataScratchSize) av1.DecoderFrameWorkSideDataScratch {
	return av1.DecoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, size.CDEFIndexMap),
		CDEFReadMap:              make([]bool, size.CDEFReadMap),
		LoopFilterMap:            make([]av1.DecoderFrameWorkLoopFilterBlockRecord, size.LoopFilterMap),
		RestorationRecords:       make([]av1.TileRestorationUnitRecord, size.RestorationRecords),
		RestorationBoundaryAbove: make([]uint16, size.RestorationBoundaryAbove),
		RestorationBoundaryBelow: make([]uint16, size.RestorationBoundaryBelow),
	}
}

func readPublicClip(t *testing.T, name string) []byte {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	path := filepath.Clean(filepath.Join(filepath.Dir(thisFile),
		"..", "internal", "av1", "testvector", "testdata", "profiles", name))
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read clip %s: %v", path, err)
	}
	return data
}

func readLibaomClip(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(libaomClipPath(name))
	if err != nil {
		t.Fatalf("read clip %s: %v", name, err)
	}
	return data
}

func libaomClipPath(name string) string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Clean(filepath.Join(filepath.Dir(thisFile),
		"..", "internal", "av1", "testdata", "libaom", name))
}

func libaomClipPresent(name string) bool {
	if _, err := os.Stat(libaomClipPath(name)); err != nil {
		return false
	}
	if _, err := os.Stat(libaomClipPath(name + ".md5")); err != nil {
		return false
	}
	return true
}
