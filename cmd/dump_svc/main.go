// Command dump_svc decodes an AV1 SVC (Scalable Video Coding) IVF stream
// using the goav1 public API and dumps reconstructed YUV samples for each
// temporal unit to stdout (or the file passed via -o).
//
// SVC streams carry multiple spatial and temporal layers in a single
// bitstream. Each AV1 OBU extension header tags its payload with a
// (TemporalID, SpatialID) pair; the AV1 sequence header's operating
// points list which combinations are valid. libaom's aomdec with the
// default output_all_layers=0 emits exactly one YUV frame per temporal
// unit at the highest SpatialID the temporal unit publishes; dump_svc
// follows the same convention by default so its output can be diffed
// against the libaom reference.
//
// Usage:
//
//	dump_svc [flags] input.ivf
//
// Flags:
//
//	-o                write YUV output to this file (default: stdout)
//	-workers          tile worker goroutine count (default 1)
//	-quiet            suppress the per-event SVC log line on stderr
//	-svc-info         only print the parsed SVC structure (sequence
//	                  operating points, scalability metadata, per-OBU
//	                  layer fan-out) and exit without decoding pixels
//	-scaled-pred      enable scaled inter-layer prediction (default
//	                  true; SVC streams effectively require it). The
//	                  flag installs GOAV1_SCALED_PRED=1 into the
//	                  process environment if it is not already set.
//	-format           output sample format: "i420" for I420 / yuv420p
//	                  (8-bit, planar 4:2:0) or "yuv420p10le" for the
//	                  canonical FFmpeg 10-bit-LE layout. When the
//	                  bitstream bit depth does not match the requested
//	                  format the CLI rejects the run rather than
//	                  silently producing garbage.
//	-i420             shorthand for -format=i420 (alias).
//	-yuv420p10le      shorthand for -format=yuv420p10le (alias).
//	-spatial-layer    select which spatial layer to emit. -1 (the
//	                  default) follows libaom's output_all_layers=0
//	                  semantics and emits the highest-SpatialID frame
//	                  the temporal unit publishes. A non-negative
//	                  value pins emission to that SpatialID; layers
//	                  the temporal unit does not publish at that ID
//	                  are skipped.
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
	"sort"
	"strings"
	"time"

	av1 "github.com/thesyncim/goav1"
)

// scaledPredEnvVar mirrors internal/av1/threading.frameWorkScaledRefAllowedEnv.
// dump_svc deals exclusively in SVC bitstreams, where the same-size-only
// gate kept by the default build rejects valid inter-layer references with
// the `threading: invalid batch` error; we therefore opt the process into
// the scaled-prediction path by default.
const scaledPredEnvVar = "GOAV1_SCALED_PRED"

// Output format identifiers. Aliases match the canonical FFmpeg names so
// callers can copy/paste filter chains.
const (
	outputFormatI420        = "i420"
	outputFormatYUV420P10LE = "yuv420p10le"
)

// spatialLayerHighest signals that the emit selector should follow libaom
// aomdec's `output_all_layers=0` convention: emit the last non-nil output
// of each temporal unit, i.e. the highest spatial layer the unit actually
// produced.
const spatialLayerHighest = -1

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
	scaledPred := fs.Bool("scaled-pred", true, "enable scaled inter-layer prediction (sets GOAV1_SCALED_PRED=1)")
	format := fs.String("format", outputFormatI420, "output sample format: i420 (8-bit) or yuv420p10le (10-bit LE)")
	useI420 := fs.Bool("i420", false, "shorthand for -format=i420")
	use10LE := fs.Bool("yuv420p10le", false, "shorthand for -format=yuv420p10le")
	spatialLayer := fs.Int("spatial-layer", spatialLayerHighest,
		"select spatial layer to emit (-1 = highest layer the temporal unit publishes)")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: dump_svc [-o out.yuv] [-workers N] [-quiet] [-svc-info]\n"+
			"                [-scaled-pred=BOOL] [-format=i420|yuv420p10le]\n"+
			"                [-i420] [-yuv420p10le] [-spatial-layer=N] input.ivf\n\n")
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

	// Resolve format aliases. The boolean shorthands override -format if
	// they are set; setting both is a configuration error.
	if *useI420 && *use10LE {
		return errors.New("cannot combine -i420 and -yuv420p10le")
	}
	switch {
	case *useI420:
		*format = outputFormatI420
	case *use10LE:
		*format = outputFormatYUV420P10LE
	}
	switch *format {
	case outputFormatI420, outputFormatYUV420P10LE:
	default:
		return fmt.Errorf("unknown -format %q (want i420 or yuv420p10le)", *format)
	}

	if *spatialLayer < spatialLayerHighest || *spatialLayer > 3 {
		return fmt.Errorf("spatial-layer must be in [-1, 3], got %d", *spatialLayer)
	}

	// Hard-wire the scaled-prediction gate on by default. SVC bitstreams
	// that ever reference a smaller spatial layer at a different
	// resolution fail with `threading: invalid batch` without it; the
	// fast (non-SVC) eight-vector cohort doesn't reach this code path so
	// flipping the global default here is safe for this CLI's audience.
	// Only override the process environment when the caller did not
	// already pin a value so we don't clobber an explicit `=0`.
	if *scaledPred {
		if os.Getenv(scaledPredEnvVar) == "" {
			if err := os.Setenv(scaledPredEnvVar, "1"); err != nil {
				return fmt.Errorf("set %s=1: %w", scaledPredEnvVar, err)
			}
		}
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

	fmt.Fprintf(stderr, "dump_svc: %s %dx%d frames=%d timebase=%d/%d format=%s spatial_layer=%d scaled_pred=%v\n",
		inputPath, header.Width, header.Height, header.FrameCount,
		header.TimebaseNum, header.TimebaseDen, *format, *spatialLayer, *scaledPred)

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

	switch *format {
	case outputFormatI420:
		if dec.format.BitDepth != 8 {
			return fmt.Errorf("format=i420 requires 8-bit input, stream is %d-bit", dec.format.BitDepth)
		}
	case outputFormatYUV420P10LE:
		if dec.format.BitDepth != 10 {
			return fmt.Errorf("format=yuv420p10le requires 10-bit input, stream is %d-bit", dec.format.BitDepth)
		}
	}

	start := time.Now()
	stats, err := dec.Decode(writer, decodeOptions{
		Quiet:        *quiet,
		Log:          stderr,
		SpatialLayer: int8(*spatialLayer),
	})
	elapsed := time.Since(start)
	if err != nil {
		fmt.Fprintf(stderr,
			"dump_svc: partial decode after %d/%d payloads (%d completed_frames, %d bytes out): %v\n",
			stats.PayloadsConsumed, len(payloads), stats.CompletedFrames, stats.BytesOut, err)
		if stats.FailureEvents != "" {
			fmt.Fprintf(stderr, "dump_svc: failing payload layer composition: %s\n", stats.FailureEvents)
		}
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

func dumpSVCStructure(frames []av1.IVFFrame, log io.Writer) error {
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
		temporalMask := uint8(op.IDC & 0xFF)
		spatialMask := uint8((op.IDC >> 8) & 0x0F)
		fmt.Fprintf(log, "dump_svc:   operating_point[%d] idc=0x%04x temporal_mask=0x%02x spatial_mask=0x%02x level_idx=%d tier=%d\n",
			i, op.IDC, temporalMask, spatialMask, op.SeqLevelIdx, op.SeqTier)
	}

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
				counts[layerKey{}]++
			}
			payload = payload[n:]
		}
	}
	if len(counts) == 0 {
		fmt.Fprintf(log, "dump_svc: no SVC-tagged OBUs (single layer stream)\n")
	} else {
		fmt.Fprintf(log, "dump_svc: per-layer OBU counts (excluding SequenceHeader/TD/Metadata):\n")
		for t := range uint8(8) {
			for s := range uint8(4) {
				if c, ok := counts[layerKey{T: t, S: s}]; ok {
					fmt.Fprintf(log, "dump_svc:   T=%d S=%d obus=%d\n", t, s, c)
				}
			}
		}
	}
	return nil
}

// describePayloadLayers walks one IVF frame payload at the OBU level and
// returns a stable "T=0 S=0,T=0 S=1" summary of every (TemporalID,
// SpatialID) pair the payload publishes. It is used by Decode() to
// annotate the proximate decoder error with which layer composition was
// in flight when the failure occurred.
func describePayloadLayers(payload []byte) string {
	type layerKey struct{ T, S uint8 }
	seen := make(map[layerKey]struct{})
	for len(payload) > 0 {
		unit, n, err := av1.ParseLowOverheadOBU(payload)
		if err != nil {
			return fmt.Sprintf("<obu parse error: %v>", err)
		}
		if unit.Header.Extension {
			seen[layerKey{T: unit.Header.TemporalID, S: unit.Header.SpatialID}] = struct{}{}
		} else if unit.Header.Type != av1.OBUSequenceHeader &&
			unit.Header.Type != av1.OBUTemporalDelimiter &&
			unit.Header.Type != av1.OBUMetadata {
			seen[layerKey{}] = struct{}{}
		}
		payload = payload[n:]
	}
	if len(seen) == 0 {
		return "(no layer-tagged OBUs)"
	}
	keys := make([]layerKey, 0, len(seen))
	for k := range seen {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].T != keys[j].T {
			return keys[i].T < keys[j].T
		}
		return keys[i].S < keys[j].S
	})
	var out strings.Builder
	for i, k := range keys {
		if i > 0 {
			out.WriteString(",")
		}
		out.WriteString(fmt.Sprintf("T=%d S=%d", k.T, k.S))
	}
	return out.String()
}

type nopWriteCloser struct{ io.Writer }

func (nopWriteCloser) Close() error { return nil }
