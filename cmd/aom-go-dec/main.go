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
//	aom-go-dec [-o out.yuv] [-workers N] [-framedelay N] [-quiet] [-cpuprofile out.pprof] input.ivf
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
	"runtime"
	"runtime/pprof"
	"time"
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
	workers := fs.Int("workers", 0, "worker goroutine count (0 = one per CPU, 1 = single-threaded)")
	frameDelay := fs.Int("framedelay", 0, "maximum frames in flight (dav1d --framedelay; 0 = auto, 1 = no frame threading)")
	quiet := fs.Bool("quiet", false, "suppress per-frame timing on stderr")
	cpuProfile := fs.String("cpuprofile", "", "write a CPU profile of the decode loop to this path")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: aom-go-dec [-o out.yuv] [-workers N] [-framedelay N] [-quiet] [-cpuprofile out.pprof] input.ivf\n\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		fs.Usage()
		return errors.New("expected exactly one input IVF path")
	}
	if *workers < 0 {
		return fmt.Errorf("workers must be >= 0, got %d", *workers)
	}
	if *workers == 0 {
		*workers = runtime.NumCPU()
	} else if *workers < runtime.GOMAXPROCS(0) {
		// Match dav1d --threads N: cap active OS threads to the requested
		// worker budget so extra Ps do not spin or spill onto efficiency cores.
		runtime.GOMAXPROCS(*workers)
	}

	inputPath := fs.Arg(0)
	input, err := os.Open(inputPath)
	if err != nil {
		return fmt.Errorf("open %q: %w", inputPath, err)
	}
	defer input.Close()
	info, err := input.Stat()
	if err != nil {
		return fmt.Errorf("stat %q: %w", inputPath, err)
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

	dec, header, err := newDecoder(input, info.Size(), *workers, *frameDelay)
	if err != nil {
		return fmt.Errorf("decoder bind: %w", err)
	}
	defer dec.Close()

	fmt.Fprintf(stderr, "aom-go-dec: %s %dx%d %d frames @ %d/%d\n",
		inputPath, header.Width, header.Height, header.FrameCount,
		header.TimebaseNum, header.TimebaseDen)

	if *cpuProfile != "" {
		pf, perr := os.Create(*cpuProfile)
		if perr != nil {
			return fmt.Errorf("open cpu profile %q: %w", *cpuProfile, perr)
		}
		defer pf.Close()
		if perr := pprof.StartCPUProfile(pf); perr != nil {
			return fmt.Errorf("start cpu profile: %w", perr)
		}
		defer pprof.StopCPUProfile()
	}

	start := time.Now()
	totalBytes, decoded, err := dec.Decode(writer, *quiet, stderr)
	if err != nil {
		return fmt.Errorf("decode: %w", err)
	}
	elapsed := time.Since(start)

	fps := float64(decoded) / elapsed.Seconds()
	mbps := float64(info.Size()) / (1024 * 1024) / elapsed.Seconds()
	fmt.Fprintf(stderr,
		"aom-go-dec: decoded %d frames in %s (%.2f fps, %.2f MB/s in, %d YUV bytes out)\n",
		decoded, elapsed.Round(time.Microsecond), fps, mbps, totalBytes)
	return nil
}

// nopWriteCloser wraps an io.Writer with a no-op Close so we can treat
// stdout and *os.File uniformly.
type nopWriteCloser struct{ io.Writer }

func (nopWriteCloser) Close() error { return nil }
