// Command aom-go-dec decodes an AV1 IVF file to raw YUV using the goav1
// public decoder API.
//
// It is intended as both a smoke-tool for the library and as executable
// documentation for callers wiring goav1 into their own decoder. The pipeline
// uses the high-level Decoder convenience API, which drives the same residual
// stream runner as the lower-level examples while also handling caller-owned
// output surfaces needed by super-res streams.
//
// Usage:
//
//	aom-go-dec [-o out.yuv] input.ivf
//
// When -o is omitted the decoded I420 (or I400 for monochrome streams) bytes
// are written to standard output, one frame after another in display order.
// Per-frame timing and aggregate throughput are reported on standard error
// so the YUV stream on stdout stays clean and can be piped into ffplay etc.
//
// The CLI uses only the public goav1 API surface; no internal packages are
// imported. It runs residual decode plus reconstruction and applies the
// loop-filter / CDEF / super-res / loop-restoration / film-grain post-filter
// chain through the same Decoder path exposed to library users. The reported
// width/height/bit depth come from the parsed sequence header.
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
		fmt.Fprintln(os.Stderr, "aom-go-dec:", err)
		os.Exit(1)
	}
}

func run(args []string, stdout io.Writer, stderr io.Writer) error {
	fs := flag.NewFlagSet("aom-go-dec", flag.ContinueOnError)
	fs.SetOutput(stderr)
	out := fs.String("o", "", "write YUV output to this file (default: stdout)")
	workers := fs.Int("workers", 1, "tile worker goroutine count")
	quiet := fs.Bool("quiet", false, "suppress per-frame timing on stderr")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: aom-go-dec [-o out.yuv] [-workers N] [-quiet] input.ivf\n\n")
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

	// Collect every IVF frame payload up-front. The public stream runner
	// accepts a slice of payloads in display order; for small/medium files
	// this is the simplest end-to-end driver. A streaming reader could push
	// payloads through RunLowOverheadInto one at a time.
	frames, header, err := collectIVFFrames(ivfBytes)
	if err != nil {
		return fmt.Errorf("ivf parse: %w", err)
	}
	if len(frames) == 0 {
		return errors.New("ivf contains no frames")
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

	dec, err := newDecoder(frames, *workers)
	if err != nil {
		return fmt.Errorf("decoder bind: %w", err)
	}
	defer dec.Close()

	fmt.Fprintf(stderr, "aom-go-dec: %s %dx%d %d frames @ %d/%d\n",
		inputPath, header.Width, header.Height, header.FrameCount,
		header.TimebaseNum, header.TimebaseDen)

	start := time.Now()
	totalBytes, decoded, err := dec.Decode(writer, *quiet, stderr)
	if err != nil {
		return fmt.Errorf("decode: %w", err)
	}
	elapsed := time.Since(start)

	fps := float64(decoded) / elapsed.Seconds()
	mbps := float64(len(ivfBytes)) / (1024 * 1024) / elapsed.Seconds()
	fmt.Fprintf(stderr,
		"aom-go-dec: decoded %d frames in %s (%.2f fps, %.2f MB/s in, %d YUV bytes out)\n",
		decoded, elapsed.Round(time.Microsecond), fps, mbps, totalBytes)
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

// nopWriteCloser wraps an io.Writer with a no-op Close so we can treat
// stdout and *os.File uniformly.
type nopWriteCloser struct{ io.Writer }

func (nopWriteCloser) Close() error { return nil }
