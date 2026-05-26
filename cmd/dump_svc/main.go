// Command dump_svc decodes an AV1 SVC (Scalable Video Coding) IVF stream
// using the goav1 public API and dumps the highest-layer reconstructed
// YUV for each temporal unit to stdout (or the file passed via -o).
//
// SVC streams carry multiple spatial and temporal layers in a single
// bitstream. Each AV1 OBU extension header tags its payload with a
// (TemporalID, SpatialID) pair; the AV1 sequence header's operating
// points list which combinations are valid. libaom's aomdec with the
// default output_all_layers=0 emits exactly one YUV frame per temporal
// unit at the highest SpatialID the temporal unit publishes; dump_svc
// follows that convention so its output can be diffed against the
// libaom reference.
//
// Usage:
//
//	dump_svc [-o out.yuv] [-workers N] [-quiet] [-svc-info] input.ivf
//
// Flags:
//
//	-o          write YUV output to this file (default: stdout)
//	-workers    tile worker goroutine count (default 1)
//	-quiet      suppress the per-event SVC log line on stderr
//	-svc-info   only print the parsed SVC structure (sequence operating
//	            points, scalability metadata, per-OBU layer fan-out)
//	            and exit without decoding pixels
//
// Limitations of the simple driver in this file:
//
//   - The single-pool stream runner used here works end-to-end only on
//     SVC streams whose spatial layers share one frame size (e.g. L1T2,
//     where only temporal layering is present). Truly multi-spatial
//     streams (L2T1, L2T2) need a per-spatial-layer frame pool wired
//     through the multi-pool decoder.FrameSurfaceProvider /
//     decoder.FrameSurfaceReleaser interfaces (see docs/svc.md).
//
//   - Decoding scaled inter-layer prediction is gated by the
//     GOAV1_SCALED_PRED=1 runtime environment variable or the
//     goav1_scaled_pred build tag. Set the variable when running this
//     command against an SVC stream whose enhancement-layer frames
//     reference base-layer surfaces at a different resolution.
//
//   - The decode pipeline does not run the post-filter chain in this
//     program — same as cmd/aom-go-dec — so emitted YUV is residual-
//     reconstructed but not deblocked/CDEFed/restored. The framework
//     dry-run tests in internal/av1/testvector exercise the full
//     post-filter chain when bit-exact MD5 parity is required.
//
//   - Decoder errors mid-stream are reported on stderr and the program
//     exits non-zero. This makes the binary safe to use as a smoke tool
//     while the SVC decode path stabilises.
//
// See docs/svc.md for a longer integrator-facing explanation of how SVC
// is signalled in AV1 and how to drive multi-pool SVC streams through
// the full goav1 public surface.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	av1 "github.com/thesyncim/goav1"
)

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "dump_svc:", err)
		os.Exit(1)
	}
}

func run(args []string, stdout io.Writer, stderr io.Writer) error {
	fs := flag.NewFlagSet("dump_svc", flag.ContinueOnError)
	fs.SetOutput(stderr)
	out := fs.String("o", "", "write YUV output to this file (default: stdout)")
	workers := fs.Int("workers", 1, "tile worker goroutine count")
	quiet := fs.Bool("quiet", false, "suppress per-event SVC log line")
	svcInfo := fs.Bool("svc-info", false, "only print SVC structure and exit")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: dump_svc [-o out.yuv] [-workers N] [-quiet] [-svc-info] input.ivf\n\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		fs.Usage()
		return errors.New("expected exactly one input IVF path")
	}
	if *workers < 1 {
		return fmt.Errorf("workers must be >= 1, got %d", *workers)
	}

	inputPath := fs.Arg(0)
	ivfBytes, err := os.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("read %q: %w", inputPath, err)
	}

	frames, header, err := collectIVFFrames(ivfBytes)
	if err != nil {
		return fmt.Errorf("ivf parse: %w", err)
	}
	if len(frames) == 0 {
		return errors.New("ivf contains no frames")
	}

	fmt.Fprintf(stderr, "dump_svc: %s %dx%d frames=%d timebase=%d/%d\n",
		inputPath, header.Width, header.Height, header.FrameCount,
		header.TimebaseNum, header.TimebaseDen)

	// Probe the stream once to extract the sequence header and the
	// per-OBU (TemporalID, SpatialID) fan-out so callers see the
	// scalability layout before any pixel work runs.
	if err := dumpSVCStructure(frames, stderr); err != nil {
		return fmt.Errorf("svc info: %w", err)
	}
	if *svcInfo {
		return nil
	}

	payloads := make([][]byte, len(frames))
	for i, f := range frames {
		payloads[i] = f.Payload
	}

	var writer io.WriteCloser
	if *out == "" {
		writer = nopWriteCloser{stdout}
	} else {
		f, ferr := os.Create(*out)
		if ferr != nil {
			return fmt.Errorf("open output %q: %w", *out, ferr)
		}
		writer = f
	}
	defer writer.Close()

	dec, err := newDecoder(payloads, *workers)
	if err != nil {
		return fmt.Errorf("decoder bind: %w", err)
	}
	defer dec.Close()

	start := time.Now()
	stats, err := dec.Decode(writer, *quiet, stderr)
	elapsed := time.Since(start)
	if err != nil {
		// Report partial decode counts before returning so callers can
		// see how far the stream made it.
		fmt.Fprintf(stderr,
			"dump_svc: partial decode after %d/%d payloads (%d completed_frames, %d bytes out): %v\n",
			stats.PayloadsConsumed, len(payloads), stats.CompletedFrames, stats.BytesOut, err)
		return err
	}

	fps := 0.0
	if elapsed > 0 {
		fps = float64(stats.CompletedFrames) / elapsed.Seconds()
	}
	fmt.Fprintf(stderr,
		"dump_svc: decoded %d temporal units, %d completed frames in %s (%.2f fps, %d YUV bytes out)\n",
		stats.TemporalUnits, stats.CompletedFrames, elapsed.Round(time.Microsecond), fps, stats.BytesOut)
	return nil
}

func collectIVFFrames(ivfBytes []byte) ([]av1.IVFFrame, av1.IVFHeader, error) {
	it, err := av1.NewIVFIterator(ivfBytes)
	if err != nil {
		return nil, av1.IVFHeader{}, err
	}
	header := it.Header()
	frames := make([]av1.IVFFrame, 0, header.FrameCount)
	for {
		frame, ok, nextErr := it.Next()
		if nextErr != nil {
			return nil, header, nextErr
		}
		if !ok {
			break
		}
		frames = append(frames, frame)
	}
	return frames, header, nil
}

// dumpSVCStructure walks the IVF payloads at the OBU level, locates the
// first sequence header, and logs the operating-points list together
// with the per-temporal-unit (TemporalID, SpatialID) fan-out so callers
// can see how the bitstream is layered before any frame work runs.
func dumpSVCStructure(frames []av1.IVFFrame, log io.Writer) error {
	// Find the first sequence header in the stream so we can log the
	// operating-points list.
	var seq av1.SequenceHeader
	haveSeq := false
SequenceScan:
	for _, frame := range frames {
		payload := frame.Payload
		for len(payload) > 0 {
			unit, n, err := av1.ParseLowOverheadOBU(payload)
			if err != nil {
				return fmt.Errorf("obu parse: %w", err)
			}
			if unit.Header.Type == av1.OBUSequenceHeader {
				seq, err = av1.ParseSequenceHeader(unit.Payload)
				if err != nil {
					return fmt.Errorf("parse sequence: %w", err)
				}
				haveSeq = true
				break SequenceScan
			}
			payload = payload[n:]
		}
	}
	if !haveSeq {
		return errors.New("ivf contains no sequence header")
	}

	fmt.Fprintf(log, "dump_svc: sequence operating_points=%d max=%dx%d order_hint_bits=%d enable_order_hint=%v\n",
		seq.OperatingPointsCount, seq.MaxFrameWidth, seq.MaxFrameHeight,
		seq.OrderHintBits, seq.EnableOrderHint)
	for i := uint8(0); i < seq.OperatingPointsCount; i++ {
		op := seq.OperatingPoints[i]
		// operating_point_idc bits 0..7 select temporal layers; bits 8..15
		// select spatial layers. An IDC of 0 means "all layers" (the
		// default single-layer profile).
		temporalMask := uint8(op.IDC & 0xFF)
		spatialMask := uint8((op.IDC >> 8) & 0x0F)
		fmt.Fprintf(log, "dump_svc:   operating_point[%d] idc=0x%04x temporal_mask=0x%02x spatial_mask=0x%02x level_idx=%d tier=%d\n",
			i, op.IDC, temporalMask, spatialMask, op.SeqLevelIdx, op.SeqTier)
	}

	// Tally how many OBUs land at each (TemporalID, SpatialID) so
	// callers can see the scalability pattern at a glance.
	type layerKey struct{ T, S uint8 }
	counts := make(map[layerKey]int)
	for _, frame := range frames {
		payload := frame.Payload
		for len(payload) > 0 {
			unit, n, err := av1.ParseLowOverheadOBU(payload)
			if err != nil {
				return fmt.Errorf("obu parse: %w", err)
			}
			if unit.Header.Extension {
				counts[layerKey{T: unit.Header.TemporalID, S: unit.Header.SpatialID}]++
			} else if unit.Header.Type != av1.OBUSequenceHeader && unit.Header.Type != av1.OBUTemporalDelimiter && unit.Header.Type != av1.OBUMetadata {
				// OBUs without extension headers belong to (T=0, S=0).
				counts[layerKey{}]++
			}
			payload = payload[n:]
		}
	}
	if len(counts) == 0 {
		fmt.Fprintf(log, "dump_svc: no SVC-tagged OBUs (single layer stream)\n")
	} else {
		fmt.Fprintf(log, "dump_svc: per-layer OBU counts (excluding SequenceHeader/TD/Metadata):\n")
		// Print in a stable order.
		for t := uint8(0); t < 8; t++ {
			for s := uint8(0); s < 4; s++ {
				if c, ok := counts[layerKey{T: t, S: s}]; ok {
					fmt.Fprintf(log, "dump_svc:   T=%d S=%d obus=%d\n", t, s, c)
				}
			}
		}
	}
	return nil
}

// nopWriteCloser wraps an io.Writer with a no-op Close so we can treat
// stdout and *os.File uniformly.
type nopWriteCloser struct{ io.Writer }

func (nopWriteCloser) Close() error { return nil }
