package main

import (
	"errors"
	"fmt"
	"io"
	"time"

	av1 "github.com/thesyncim/goav1"
)

// decoder owns the public high-level decoder used by the CLI. The high-level
// path still drives the residual stream runner, but it also selects the
// caller-owned postfilter publication path when superres needs a detached
// upscaled reference surface.
type decoder struct {
	dec *av1.Decoder
}

func newDecoder(src io.ReaderAt, size int64, workers int) (*decoder, av1.IVFHeader, error) {
	if workers < 1 {
		return nil, av1.IVFHeader{}, fmt.Errorf("workers must be >= 1, got %d", workers)
	}
	dec, header, err := av1.NewDecoderFromIVFReaderAt(src, size, av1.WithWorkers(workers))
	if err != nil {
		return nil, header, err
	}
	return &decoder{dec: dec}, header, nil
}

// Close releases the worker goroutine pool. It is safe to call more than once.
func (d *decoder) Close() {
	if d.dec != nil {
		d.dec.Close()
		d.dec = nil
	}
}

// Decode iterates the bound payloads, drives them through the public decoder one
// payload at a time so per-frame timing is meaningful, writes each visible
// output frame's planes to dst in display order, and reports total bytes written
// plus visible-frame count.
func (d *decoder) Decode(dst io.Writer, quiet bool, log io.Writer) (int64, int, error) {
	if d == nil || d.dec == nil {
		return 0, 0, errors.New("decoder closed")
	}
	if err := d.dec.Reset(); err != nil {
		return 0, 0, fmt.Errorf("decoder reset: %w", err)
	}

	var totalBytes int64
	completed := 0
	for i := 0; ; i++ {
		start := time.Now()
		frames, ok, err := d.dec.DecodeNext()
		if err != nil {
			return totalBytes, completed, fmt.Errorf("frame %d: %w", i, err)
		}
		if !ok {
			break
		}
		elapsed := time.Since(start)

		for _, frame := range frames {
			if frame == nil {
				continue
			}
			n, err := writeYUVFrame(dst, frame)
			if err != nil {
				return totalBytes, completed, fmt.Errorf("frame %d write: %w", i, err)
			}
			totalBytes += n
			completed++
		}
		if !quiet {
			fmt.Fprintf(log, "frame %d ms=%.3f outputs=%d completed_frames=%d\n",
				i, float64(elapsed.Microseconds())/1000.0, len(frames), completed)
		}
	}
	return totalBytes, completed, nil
}

// writeYUVFrame writes a decoded frame's Y, U, V planes to dst stripping any
// per-plane stride padding. For monochrome streams only Y is written. 10/12-bit
// samples are written little-endian as two bytes per sample, matching the
// canonical FFmpeg "yuv420p10le" layout.
func writeYUVFrame(dst io.Writer, frame *av1.Frame) (int64, error) {
	var written int64
	for _, plane := range []av1.FramePlane{frame.Y, frame.U, frame.V} {
		// MonoChrome streams leave U/V zero-sized; skip them so the output
		// matches the canonical I400 byte layout.
		if plane.Width == 0 || plane.Height == 0 || len(plane.Pix) == 0 {
			continue
		}
		rowBytes := plane.Width * frame.Layout.BytesPerSample
		for row := 0; row < plane.Height; row++ {
			off := row * plane.Stride
			n, err := dst.Write(plane.Pix[off : off+rowBytes])
			if err != nil {
				return written + int64(n), err
			}
			written += int64(n)
		}
	}
	return written, nil
}
