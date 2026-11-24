package goav1

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"runtime"
	"runtime/metrics"
	"slices"
	"strings"
	"testing"
)

// The GC-invisibility gate: once warm, decode (1 and 8 workers, replayed
// after Reset) and encode (MaxThreads 1 and 0) must allocate nothing from
// goav1 code, so the codec never paces or assists the application's garbage
// collector. Every allocation in the measured window is profiled
// (MemProfileRate 1) and classified by its stack:
//
//   - goav1 non-test code: a codec allocation, which fails the gate;
//   - runtime.acquireSudog: the runtime refilling its sudog cache when a
//     worker parks on a channel, mutex or condition variable. Each GC empties
//     the central cache, so a few of these follow every collection whatever
//     the codec does; they are reported, not gated;
//   - anything else (test code, runtime internals such as new Ms and timers)
//     is ignored.
//
// The streams are generated in-test so the gate runs without the local-only
// corpora, at a size that engages the decoder's frame-thread pipeline and
// SB-row wavefront and the encoder's wavefront lanes.

type gcInvisibleAllocs struct {
	codec  []string
	sudogs int64
	other  int64 // test code, the profile snapshot, runtime internals
}

// gcInvisibleProfile snapshots the cumulative allocation counts of every
// profiled stack. Profile records are published at the end of a GC cycle, so
// it collects first; the snapshot's own allocations are attributed to this
// function and filtered out by the classifier.
func gcInvisibleProfile() map[[32]uintptr]int64 {
	for range 3 {
		runtime.GC()
	}
	n, _ := runtime.MemProfile(nil, true)
	recs := make([]runtime.MemProfileRecord, n+512)
	n, _ = runtime.MemProfile(recs, true)
	out := make(map[[32]uintptr]int64, n)
	for i := range recs[:n] {
		out[recs[i].Stack0] += recs[i].AllocObjects
	}
	return out
}

// measureCodecAllocs runs fn with every allocation profiled and returns the
// allocation sites attributed to goav1 code.
func measureCodecAllocs(t *testing.T, fn func()) gcInvisibleAllocs {
	t.Helper()
	defer func(rate int) { runtime.MemProfileRate = rate }(runtime.MemProfileRate)
	runtime.MemProfileRate = 1
	before := gcInvisibleProfile()
	fn()
	after := gcInvisibleProfile()
	var res gcInvisibleAllocs
	for stack, count := range after {
		d := count - before[stack]
		if d <= 0 {
			continue
		}
		kind, desc := classifyAllocStack(stack)
		switch kind {
		case allocCodec:
			res.codec = append(res.codec, fmt.Sprintf("%d objects at\n%s", d, desc))
		case allocSudog:
			res.sudogs += d
		default:
			res.other += d
		}
	}
	return res
}

const (
	allocOther = iota
	allocCodec
	allocSudog
)

func classifyAllocStack(stack [32]uintptr) (int, string) {
	n := 0
	for n < len(stack) && stack[n] != 0 {
		n++
	}
	frames := runtime.CallersFrames(stack[:n])
	var desc strings.Builder
	kind := allocOther
	decided := false
	for {
		f, more := frames.Next()
		fmt.Fprintf(&desc, "    %s\n        %s:%d\n", f.Function, f.File, f.Line)
		switch {
		case f.Function == "runtime.acquireSudog":
			return allocSudog, ""
		case decided:
		case strings.HasPrefix(f.Function, "github.com/thesyncim/goav1"):
			decided = true
			if strings.Contains(f.Function, "gcInvisibleProfile") {
				return allocOther, ""
			}
			if !strings.HasSuffix(f.File, "_test.go") {
				kind = allocCodec
			}
		}
		if !more {
			break
		}
	}
	return kind, desc.String()
}

func (a gcInvisibleAllocs) check(t *testing.T, what string) {
	t.Helper()
	t.Logf("%s: %d codec allocation sites, %d runtime sudog refills, %d other (test/runtime) objects", what, len(a.codec), a.sudogs, a.other)
	for _, site := range a.codec {
		t.Errorf("%s allocated %s", what, site)
	}
}

const gcInvisibleW, gcInvisibleH = 640, 360

// gcInvisibleStream encodes a short moving 640x360 stream with periodic
// keyframes and returns its temporal units.
func gcInvisibleStream(t testing.TB) [][]byte {
	t.Helper()
	enc, err := NewVideoEncoder(VideoEncoderConfig{
		Width: gcInvisibleW, Height: gcInvisibleH, TargetBitrate: 1_500_000, Framerate: 30, MaxThreads: 1,
	})
	if err != nil {
		t.Fatalf("NewVideoEncoder: %v", err)
	}
	defer enc.Close()
	frame := gcInvisibleFrame()
	var tus [][]byte
	for n := range 24 {
		fillMovingI420(frame, n)
		out, err := enc.Encode(frame, n%12 == 0)
		if err != nil {
			t.Fatalf("Encode %d: %v", n, err)
		}
		tus = append(tus, append([]byte(nil), out.Data...))
	}
	return tus
}

func gcInvisibleFrame() I420Frame {
	w, h := gcInvisibleW, gcInvisibleH
	return I420Frame{
		Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
		YStride: w, ChromaStride: w / 2, Width: w, Height: h,
	}
}

func TestGCInvisibleDecodeSteadyState(t *testing.T) {
	tus := gcInvisibleStream(t)
	for _, workers := range []int{1, 8} {
		t.Run(fmt.Sprintf("workers=%d", workers), func(t *testing.T) {
			dec, err := NewDecoder(tus, WithWorkers(workers))
			if err != nil {
				t.Fatalf("NewDecoder: %v", err)
			}
			defer dec.Close()
			if workers > 1 && dec.fp == nil {
				t.Fatal("frame-thread pipeline not engaged")
			}
			decodeAll := func() {
				if err := dec.Reset(); err != nil {
					t.Fatalf("Reset: %v", err)
				}
				frames := 0
				for {
					out, ok, err := dec.DecodeNext()
					if err != nil {
						t.Fatalf("DecodeNext: %v", err)
					}
					if !ok {
						break
					}
					frames += len(out)
				}
				if frames != len(tus) {
					t.Fatalf("decoded %d frames, want %d", frames, len(tus))
				}
			}
			for range 2 {
				decodeAll()
			}
			measureCodecAllocs(t, func() {
				for range 2 {
					decodeAll()
				}
			}).check(t, "decode")
		})
	}
}

// TestGCInvisibleRTPDecodeSteadyState covers the live RTP receive path, which
// runs the external post-filter runner instead of the frame pipeline.
func TestGCInvisibleRTPDecodeSteadyState(t *testing.T) {
	enc, err := NewRTCEncoderWithConfig(publicRTCMatrixConfig(gcInvisibleW, gcInvisibleH, EncoderScalabilityModeL1T3))
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	frame := gcInvisibleFrame()
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 1200}
	var payloads [][]byte
	for n := range 24 {
		fillMovingI420(frame, n)
		out, err := enc.Encode(frame, n == 12)
		if err != nil {
			t.Fatalf("Encode %d: %v", n, err)
		}
		payloads = append(payloads, publicDecoderRTPPayloadsForFrameWithLimits(t, out, limits)...)
	}
	_ = enc.Close()
	for _, workers := range []int{1, 8} {
		t.Run(fmt.Sprintf("workers=%d", workers), func(t *testing.T) {
			dec, err := NewDecoderFromRTPPayloads(payloads, WithWorkers(workers))
			if err != nil {
				t.Fatalf("NewDecoderFromRTPPayloads: %v", err)
			}
			defer dec.Close()
			decodeAll := func() {
				if err := dec.Reset(); err != nil {
					t.Fatalf("Reset: %v", err)
				}
				for i, payload := range payloads {
					if _, err := dec.DecodeRTPPayload(payload); err != nil {
						t.Fatalf("DecodeRTPPayload %d: %v", i, err)
					}
				}
			}
			for range 2 {
				decodeAll()
			}
			measureCodecAllocs(t, func() {
				for range 2 {
					decodeAll()
				}
			}).check(t, "RTP decode")
		})
	}
}

func TestGCInvisibleEncodeSteadyState(t *testing.T) {
	// MaxThreads=4 pins wavefront lanes whatever the runner's CPU count.
	for _, threads := range []int{1, 0, 4} {
		t.Run(fmt.Sprintf("MaxThreads=%d", threads), func(t *testing.T) {
			enc, err := NewVideoEncoder(VideoEncoderConfig{
				Width: gcInvisibleW, Height: gcInvisibleH, TargetBitrate: 1_500_000, Framerate: 30, MaxThreads: threads,
			})
			if err != nil {
				t.Fatalf("NewVideoEncoder: %v", err)
			}
			defer enc.Close()
			frame := gcInvisibleFrame()
			n := 0
			encode := func(frames int) {
				for range frames {
					fillMovingI420(frame, n)
					if _, err := enc.Encode(frame, n%13 == 0); err != nil {
						t.Fatalf("Encode %d: %v", n, err)
					}
					n++
				}
			}
			encode(40)
			measureCodecAllocs(t, func() { encode(40) }).check(t, "encode")
		})
	}
}

// TestGCInvisibleScannableHeap bounds the pointerful (GC-scanned) heap a warm
// codec keeps alive. Large scratch must stay pointer-free or keep its
// pointers in a short leading run. The measured footprint is 0.24, 0.79 and
// 0.65 MiB (1-worker decode, 8-worker decode, encoder). The 8-worker ceiling
// sits below the 1.49 MiB measured while the buffered block visits still
// carried payload pointers, so it catches pointers coming back into them; the
// others carry about 2x headroom (origin/main before this gate measured
// 3.7 MiB at 8 workers and 1.9 MiB for the encoder). A reading below the
// floor means the measurement was disturbed, and fails rather than passing
// vacuously.
//
// The scannable heap is a process-wide metric, so other tests in the package
// (parallel ones especially) inflate it. The check therefore re-runs itself in
// a child test process where nothing else is alive.
func TestGCInvisibleScannableHeap(t *testing.T) {
	const isolatedEnv = "GOAV1_GC_INVISIBLE_ISOLATED"
	if os.Getenv(isolatedEnv) != "1" {
		if testing.Short() {
			t.Skip("spawns a child test process")
		}
		exe, err := os.Executable()
		if err != nil {
			t.Fatalf("os.Executable: %v", err)
		}
		cmd := exec.Command(exe, "-test.run=^TestGCInvisibleScannableHeap$", "-test.count=1", "-test.v")
		cmd.Env = append(os.Environ(), isolatedEnv+"=1")
		out, err := cmd.CombinedOutput()
		t.Logf("isolated run:\n%s", out)
		if err != nil {
			t.Fatalf("isolated scannable-heap check failed: %v", err)
		}
		return
	}
	scan := func() int64 {
		runtime.GC()
		runtime.GC()
		s := []metrics.Sample{{Name: "/gc/scan/heap:bytes"}}
		metrics.Read(s)
		return int64(s[0].Value.Uint64())
	}
	tus := gcInvisibleStream(t)
	for _, tc := range []struct {
		name    string
		ceiling int64
		build   func() (func(), func())
	}{
		{"decode/workers=1", 512 << 10, func() (func(), func()) {
			dec, err := NewDecoder(tus, WithWorkers(1))
			if err != nil {
				t.Fatalf("NewDecoder: %v", err)
			}
			return func() { decodeAllFrames(t, dec) }, dec.Close
		}},
		{"decode/workers=8", 1280 << 10, func() (func(), func()) {
			dec, err := NewDecoder(tus, WithWorkers(8))
			if err != nil {
				t.Fatalf("NewDecoder: %v", err)
			}
			return func() { decodeAllFrames(t, dec) }, dec.Close
		}},
		{"encode/MaxThreads=1", 1280 << 10, func() (func(), func()) {
			enc, err := NewVideoEncoder(VideoEncoderConfig{
				Width: gcInvisibleW, Height: gcInvisibleH, TargetBitrate: 1_500_000, Framerate: 30, MaxThreads: 1,
			})
			if err != nil {
				t.Fatalf("NewVideoEncoder: %v", err)
			}
			frame := gcInvisibleFrame()
			return func() {
				for n := range 16 {
					fillMovingI420(frame, n)
					if _, err := enc.Encode(frame, n == 0); err != nil {
						t.Fatalf("Encode: %v", err)
					}
				}
			}, func() { _ = enc.Close() }
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The codec's share is what appears with it alive and goes away
			// once it is closed and dropped. Other tests in the same process
			// can leave live or churning heap behind, so take the smaller of
			// the two deltas, and the best of three attempts.
			const floor = 64 << 10
			got := int64(math.MaxInt64)
			for range 3 {
				base := scan()
				run, closeFn := tc.build()
				run()
				with := scan()
				closeFn()
				run, closeFn = nil, nil
				if d := min(with-base, with-scan()); d >= floor {
					got = min(got, d)
				}
				if got <= tc.ceiling {
					break
				}
			}
			if got == math.MaxInt64 {
				t.Fatalf("every scannable-heap reading fell below the %d-byte floor: measurement disturbed", floor)
			}
			t.Logf("scannable heap %.2f MiB (ceiling %.2f MiB)", float64(got)/(1<<20), float64(tc.ceiling)/(1<<20))
			if got < floor {
				t.Fatalf("scannable heap %d bytes, below the %d-byte floor: measurement disturbed", got, floor)
			}
			if got > tc.ceiling {
				t.Fatalf("scannable heap %d bytes, ceiling %d", got, tc.ceiling)
			}
		})
	}
}

// The park-allocation gate. A codec goroutine that parks (channel, mutex,
// WaitGroup, condition variable) takes a sudog from its P's cache. When that
// cache and the central list are both empty, runtime.acquireSudog allocates
// one, and that allocation pays the application's sweep debt: under
// GOMEMLIMIT the first allocation after a collection can sweep most of the
// heap, a 5-330 ms stall charged to the codec. Spawning a goroutine when no
// dead one is cached allocates a g (runtime.malg) the same way. Each GC
// empties the central sudog list, so the test collects between every frame
// and counts every allocation (MemProfileRate 1) whose stack holds
// acquireSudog or malg together with a goav1 frame, by the innermost goav1
// frame. parkAllocTolerated lists the functions (as parkAllocFunc prints
// them) still allowed to park-allocate; any other site fails.
//
// A P's cache runs dry because a parked goroutine usually resumes on another
// P (about 80% of releases), so Ps that park more than they resume drain.
// The allocations land on whichever goroutine parks on a drained P: removing
// some park sites moves them to the parks that remain rather than removing
// them. Sites leave the list as the scheduler phases remove them.
var parkAllocTolerated = map[string]bool{}

// parkAllocReportOnly keeps the gate reporting while the scheduler work is in
// progress.
const parkAllocReportOnly = true

type parkAllocs struct {
	sites map[string]int64 // "sudog <site>" / "malg <site>" -> objects
	gcs   int
}

func (a parkAllocs) total() (n int64) {
	for _, v := range a.sites {
		n += v
	}
	return n
}

// measureParkAllocs runs step n times with a forced collection before each
// and returns the park allocations attributed to goav1 frames. isSite picks
// the frames that count (goav1 non-test code for the codec gate).
func measureParkAllocs(n int, step func(), isSite func(runtime.Frame) bool) parkAllocs {
	defer func(rate int) { runtime.MemProfileRate = rate }(runtime.MemProfileRate)
	runtime.MemProfileRate = 1
	before := gcInvisibleProfile()
	for range n {
		runtime.GC()
		step()
	}
	after := gcInvisibleProfile()
	res := parkAllocs{sites: map[string]int64{}, gcs: n}
	for stack, count := range after {
		d := count - before[stack]
		if d <= 0 {
			continue
		}
		if kind, site := classifyParkAlloc(stack, isSite); kind != "" {
			res.sites[kind+" "+site] += d
		}
	}
	return res
}

// classifyParkAlloc returns "sudog" or "malg" and the innermost frame
// accepted by isSite, or "" when the stack is not a park allocation from an
// accepted frame.
func classifyParkAlloc(stack [32]uintptr, isSite func(runtime.Frame) bool) (string, string) {
	n := 0
	for n < len(stack) && stack[n] != 0 {
		n++
	}
	frames := runtime.CallersFrames(stack[:n])
	kind := ""
	for {
		f, more := frames.Next()
		switch {
		case kind == "" && f.Function == "runtime.acquireSudog":
			kind = "sudog"
		case kind == "" && f.Function == "runtime.malg":
			kind = "malg"
		case kind == "malg" && f.Function == "runtime.allocm":
			return "", "" // a new M's g0, not a spawned goroutine
		case kind != "" && isSite(f):
			return kind, fmt.Sprintf("%s (%s:%d)", parkAllocFunc(f.Function),
				f.File[strings.LastIndex(f.File, "/")+1:], f.Line)
		}
		if !more {
			break
		}
	}
	if kind == "malg" {
		// newproc allocates on the system stack, whose traceback ends at
		// the stack switch: the spawning frame is not recorded. Only the
		// codec spawns goroutines in the measured window (a no-op window
		// with a decoder alive records none once new Ms are excluded).
		return kind, "(system stack: spawner not recorded)"
	}
	return "", ""
}

// parkAllocFunc shortens a goav1 function name to its package-relative form
// (goav1.(*Decoder).x, decoder.(*T).y).
func parkAllocFunc(fn string) string {
	fn = strings.TrimPrefix(fn, "github.com/thesyncim/")
	return strings.TrimPrefix(fn, "goav1/internal/av1/")
}

func isCodecFrame(f runtime.Frame) bool {
	return strings.HasPrefix(f.Function, "github.com/thesyncim/goav1") && !strings.HasSuffix(f.File, "_test.go") &&
		!parkPrimitives[parkAllocFunc(f.Function)]
}

// parkPrimitives are the codec's shared wait primitives: a park inside one is
// attributed to the frame that called it.
var parkPrimitives = map[string]bool{
	"threading.(*WaitList).Park":                         true,
	"threading.(*FrameProgress).Wait":                    true,
	"threading.(*frameWorkReconWavefront).waitRows":      true,
	"encoder.(*pframeWavefront).waitRowProgress":         true,
	"decoder.(*FrameWorkPostFilterParallel).rowPipeHelp": true,
}

func (a parkAllocs) check(t *testing.T, what string) {
	t.Helper()
	keys := make([]string, 0, len(a.sites))
	for k := range a.sites {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	var b strings.Builder
	for _, k := range keys {
		fmt.Fprintf(&b, "\n  %6d  %s", a.sites[k], k)
	}
	t.Logf("%s: %d park allocations over %d collections (%.2f per collection)%s", what, a.total(), a.gcs,
		float64(a.total())/float64(a.gcs), b.String())
	if parkAllocReportOnly {
		return
	}
	for _, k := range keys {
		fn := k[strings.IndexByte(k, ' ')+1:]
		if i := strings.Index(fn, " ("); i >= 0 {
			fn = fn[:i]
		}
		if !parkAllocTolerated[fn] {
			t.Errorf("%s: park allocation at %s (%d objects)", what, k, a.sites[k])
		}
	}
}

func TestGCInvisibleParkAllocs(t *testing.T) {
	if testing.Short() {
		t.Skip("collects between every frame")
	}
	t.Run("decode/workers=8", func(t *testing.T) {
		tus := gcInvisibleStream(t)
		dec, err := NewDecoder(tus, WithWorkers(8))
		if err != nil {
			t.Fatalf("NewDecoder: %v", err)
		}
		defer dec.Close()
		if dec.fp == nil {
			t.Fatal("frame-thread pipeline not engaged")
		}
		for range 2 {
			decodeAllFrames(t, dec)
		}
		if err := dec.Reset(); err != nil {
			t.Fatalf("Reset: %v", err)
		}
		step := func() {
			for {
				_, ok, err := dec.DecodeNext()
				if err != nil {
					t.Fatalf("DecodeNext: %v", err)
				}
				if ok {
					return
				}
				if err := dec.Reset(); err != nil {
					t.Fatalf("Reset: %v", err)
				}
			}
		}
		measureParkAllocs(4*len(tus), step, isCodecFrame).check(t, "decode")
	})
	t.Run("encode/MaxThreads=0", func(t *testing.T) {
		enc, err := NewVideoEncoder(VideoEncoderConfig{
			Width: gcInvisibleW, Height: gcInvisibleH, TargetBitrate: 1_500_000, Framerate: 30, MaxThreads: 0,
		})
		if err != nil {
			t.Fatalf("NewVideoEncoder: %v", err)
		}
		defer enc.Close()
		frame := gcInvisibleFrame()
		n := 0
		step := func() {
			fillMovingI420(frame, n)
			if _, err := enc.Encode(frame, n%13 == 0); err != nil {
				t.Fatalf("Encode %d: %v", n, err)
			}
			n++
		}
		for range 40 {
			step()
		}
		measureParkAllocs(96, step, isCodecFrame).check(t, "encode")
	})
}

func decodeAllFrames(t *testing.T, dec *Decoder) {
	t.Helper()
	if err := dec.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	for {
		_, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("DecodeNext: %v", err)
		}
		if !ok {
			return
		}
	}
}
