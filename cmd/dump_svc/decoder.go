package main

import (
	"fmt"
	"io"
	"time"

	av1 "github.com/thesyncim/goav1"
)

// decodeStats accumulates the numbers dump_svc prints to stderr at the end of
// a successful run, plus the partial counts run() reports when a payload
// mid-stream fails to decode.
type decodeStats struct {
	Payloads        int
	CompletedFrames int
	BytesOut        int64

	// FailureEvents is the (TemporalID, SpatialID) layer composition of the
	// payload that triggered a decode or write error.
	FailureEvents string
}

// decode drives dec one temporal unit at a time and writes one frame per
// temporal unit to dst. With spatialLayer == spatialLayerHighest it emits the
// highest spatial layer the temporal unit published (libaom aomdec's
// output_all_layers=0); otherwise it emits the frame at that SpatialID and
// writes nothing for a temporal unit that did not publish it. The frames alias
// decoder-owned surfaces, so each is written before the next DecodeNext call.
func decode(dec *av1.LayeredDecoder, payloads [][]byte, dst, log io.Writer, quiet bool, spatialLayer int) (decodeStats, error) {
	var stats decodeStats
	for {
		start := time.Now()
		frames, ok, err := dec.DecodeNextWithMetadata()
		if err != nil {
			stats.FailureEvents = describePayloadLayers(payloads[stats.Payloads])
			return stats, fmt.Errorf("%w (%d bytes, layers=%s)", err, len(payloads[stats.Payloads]), stats.FailureEvents)
		}
		if !ok {
			return stats, nil
		}
		elapsed := time.Since(start)

		i := stats.Payloads
		stats.Payloads++
		stats.CompletedFrames += len(frames)
		emit, emitSpatialID := selectFrame(frames, spatialLayer)
		if emit != nil {
			n, err := writeYUVFrame(dst, emit)
			stats.BytesOut += n
			if err != nil {
				stats.FailureEvents = describePayloadLayers(payloads[i])
				return stats, fmt.Errorf("payload %d write: %w", i, err)
			}
		}
		if !quiet {
			fmt.Fprintf(log, "tu=%d payload=%d ms=%.3f outputs=%d emit=%v emit_spatial_id=%d\n",
				i, len(payloads[i]), float64(elapsed.Microseconds())/1000.0,
				len(frames), emit != nil, emitSpatialID)
		}
	}
}

// selectFrame picks the frame to emit for one temporal unit and its SpatialID,
// or nil and -1 when the pinned layer was not published. Frames arrive in OBU
// order, which is ascending SpatialID, so the last match is the highest layer.
func selectFrame(frames []av1.LayeredFrame, spatialLayer int) (*av1.Frame, int) {
	for i := len(frames) - 1; i >= 0; i-- {
		if spatialLayer == spatialLayerHighest || int(frames[i].SpatialID) == spatialLayer {
			return frames[i].Frame, int(frames[i].SpatialID)
		}
	}
	return nil, -1
}

// writeYUVFrame writes a decoded frame's Y, U, V planes to dst stripping
// any per-plane stride padding. For monochrome streams only Y is
// written. 10/12-bit samples are written little-endian as two bytes per
// sample, matching the canonical FFmpeg "yuv420p10le" layout.
func writeYUVFrame(dst io.Writer, frame *av1.Frame) (int64, error) {
	var written int64
	for _, plane := range []av1.FramePlane{frame.Y, frame.U, frame.V} {
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
