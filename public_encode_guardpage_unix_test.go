// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix

package goav1_test

import (
	"bytes"
	"fmt"
	"runtime/debug"
	"testing"

	av1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/guardpage"
)

// guardedI420 returns frame n of a moving test pattern whose planes are each
// exactly (rows-1)*stride+width bytes, against a guard page on the trailing
// (or, with before, the leading) side.
func guardedI420(t *testing.T, w, h, yStride, cStride, n int, before bool) av1.I420Frame {
	cw, ch := (w+1)/2, (h+1)/2
	alloc := func(rows, stride, width int) []byte {
		if before {
			return guardpage.Before[byte](t, (rows-1)*stride+width)
		}
		return guardpage.After[byte](t, (rows-1)*stride+width)
	}
	f := av1.I420Frame{
		Y: alloc(h, yStride, w), U: alloc(ch, cStride, cw), V: alloc(ch, cStride, cw),
		YStride: yStride, ChromaStride: cStride, Width: w, Height: h,
	}
	for y := range h {
		for x := range w {
			f.Y[y*yStride+x] = uint8(40 + ((x+3*n)*7+(y+n)*5)%180 + (x*y)%13)
		}
	}
	for y := range ch {
		for x := range cw {
			f.U[y*cStride+x] = uint8(110 + (x+y+n)%23)
			f.V[y*cStride+x] = uint8(140 - (2*x+y+n)%19)
		}
	}
	return f
}

// TestPublicEncodeGuardedSourcePlanes encodes keyframes and inter frames from
// source planes that end exactly at (or start right after) an inaccessible
// page, with stride equal to the width and wider than it, through
// VideoEncoder (flat and default temporal layering, one lane and automatic
// threads) and RTCEncoder. Sizes the encoder rejects must return an error
// without touching memory outside the planes.
func TestPublicEncodeGuardedSourcePlanes(t *testing.T) {
	sizes := [][2]int{{2, 2}, {17, 9}, {16, 16}, {18, 16}, {130, 66}, {366, 206}, {1280, 720}}
	for _, sz := range sizes {
		w, h := sz[0], sz[1]
		frames := 6
		if w*h > 640*360 {
			frames = 3
		}
		for _, wide := range []bool{false, true} {
			yStride, cStride := w, (w+1)/2
			if wide {
				yStride, cStride = w+40, (w+1)/2+24
			}
			for _, before := range []bool{false, true} {
				for _, cfg := range []struct {
					name    string
					layers  int
					threads int
					rtc     bool
				}{
					{"flat-1t", 1, 1, false},
					{"layered-auto", 0, 0, false},
					{"rtc", 0, 0, true},
				} {
					if wide && cfg.name != "layered-auto" {
						continue
					}
					name := fmt.Sprintf("%dx%d wide=%v before=%v %s", w, h, wide, before, cfg.name)
					encodeGuarded(t, name, w, h, yStride, cStride, before, frames, cfg.layers, cfg.threads, cfg.rtc)
				}
			}
		}
	}
}

func encodeGuarded(t *testing.T, name string, w, h, yStride, cStride int, before bool, frames, layers, threads int, rtc bool) {
	t.Helper()
	defer debug.SetPanicOnFault(debug.SetPanicOnFault(true))
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("%s: encoder panicked: %v", name, r)
		}
	}()
	cfg := av1.VideoEncoderConfig{Width: w, Height: h, TargetBitrate: 400_000, Framerate: 30, TemporalLayers: layers, MaxThreads: threads}
	var encode func(av1.I420Frame, bool) ([]byte, error)
	if rtc {
		e, err := av1.NewRTCEncoder(cfg)
		if err != nil {
			return
		}
		encode = func(f av1.I420Frame, key bool) ([]byte, error) {
			out, err := e.Encode(f, key)
			return out.Data, err
		}
	} else {
		e, err := av1.NewVideoEncoder(cfg)
		if err != nil {
			return
		}
		encode = func(f av1.I420Frame, key bool) ([]byte, error) {
			out, err := e.Encode(f, key)
			return out.Data, err
		}
	}
	var stream [][]byte
	for n := range frames {
		data, err := encode(guardedI420(t, w, h, yStride, cStride, n, before), n == 0)
		if err != nil {
			t.Fatalf("%s: frame %d: %v", name, n, err)
		}
		stream = append(stream, bytes.Clone(data))
	}
	if len(stream[0]) == 0 {
		t.Fatalf("%s: empty keyframe", name)
	}
}
