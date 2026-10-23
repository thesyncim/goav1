// Command decoder is the goav1 AV1 decoder compiled to WebAssembly. Unlike
// the encoder it talks to JS through syscall/js, so the same source builds
// with both TinyGo (-target=wasm) and stock Go (GOOS=js GOARCH=wasm) and the
// two toolchains can be compared in the same browser.
//
// It installs a global `goav1dec` object in the worker:
//
//	open(ivf Uint8Array) -> error string ("" on success)
//	next()               -> {width, height, data: Uint8Array I420} or null at end
//	reset()              -> error string
//	bench(passes)        -> {frames, msPerFrame, md5} (best pass; md5 of the
//	                        visible I420 output, aomdec --md5 convention)
//
// Frames wider than 8 bits are shifted down to 8-bit for display.
package main

import (
	"crypto/md5"
	"encoding/hex"
	"runtime"
	"syscall/js"
	"time"

	goav1 "github.com/thesyncim/goav1"
)

var (
	dec     *goav1.Decoder
	pending []*goav1.Frame // frames from the last DecodeNext not yet handed out
	packed  []byte
)

func main() {
	api := js.Global().Get("Object").New()
	api.Set("open", js.FuncOf(open))
	api.Set("next", js.FuncOf(next))
	api.Set("reset", js.FuncOf(reset))
	api.Set("bench", js.FuncOf(bench))
	js.Global().Set("goav1dec", api)
	select {}
}

func open(_ js.Value, args []js.Value) any {
	if dec != nil {
		dec.Close()
		dec = nil
	}
	ivf := make([]byte, args[0].Get("byteLength").Int())
	js.CopyBytesToGo(ivf, args[0])
	d, err := goav1.NewDecoderFromIVF(ivf, goav1.WithWorkers(1))
	if err != nil {
		return err.Error()
	}
	dec, pending = d, nil
	return ""
}

func reset(js.Value, []js.Value) any {
	pending = nil
	if err := dec.Reset(); err != nil {
		return err.Error()
	}
	return ""
}

// next returns the next visible frame as packed 8-bit I420, decoding further
// temporal units as needed (some produce no visible frame, some several).
func next(js.Value, []js.Value) any {
	for len(pending) == 0 {
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			return js.Global().Get("Error").New(err.Error())
		}
		if !ok {
			return nil
		}
		pending = frames
	}
	f := pending[0]
	pending = pending[1:]
	packed = pack8(packed[:0], f)
	data := js.Global().Get("Uint8Array").New(len(packed))
	js.CopyBytesToJS(data, packed)
	return map[string]any{"width": f.Y.Width, "height": f.Y.Height, "data": data}
}

// pack8 appends the visible Y, U, V samples of f without stride padding,
// reducing high-bit-depth samples to their top 8 bits.
func pack8(dst []byte, f *goav1.Frame) []byte {
	shift := uint(f.Format.BitDepth) - 8
	for _, p := range []goav1.FramePlane{f.Y, f.U, f.V} {
		for r := 0; r < p.Height; r++ {
			row := p.Pix[r*p.Stride:]
			if f.Layout.BytesPerSample == 1 {
				dst = append(dst, row[:p.Width]...)
				continue
			}
			for x := 0; x < p.Width; x++ {
				dst = append(dst, byte((uint16(row[2*x])|uint16(row[2*x+1])<<8)>>shift))
			}
		}
	}
	return dst
}

// bench decodes the whole stream `passes` times, timing only DecodeNext, then
// once more hashing the output so the result can be checked against the
// corpus .md5 files.
func bench(_ js.Value, args []js.Value) any {
	passes := args[0].Int()
	best := time.Duration(1 << 62)
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	frames := 0
	for p := 0; p < passes; p++ {
		if err := dec.Reset(); err != nil {
			return js.Global().Get("Error").New(err.Error())
		}
		frames = 0
		var total time.Duration
		for {
			t0 := time.Now()
			out, ok, err := dec.DecodeNext()
			total += time.Since(t0)
			if err != nil {
				return js.Global().Get("Error").New(err.Error())
			}
			if !ok {
				break
			}
			frames += len(out)
		}
		best = min(best, total)
	}
	runtime.ReadMemStats(&after)
	decoded := max(frames*passes, 1)
	if err := dec.Reset(); err != nil {
		return js.Global().Get("Error").New(err.Error())
	}
	h := md5.New()
	for {
		out, ok, err := dec.DecodeNext()
		if err != nil {
			return js.Global().Get("Error").New(err.Error())
		}
		if !ok {
			break
		}
		for _, f := range out {
			bps := f.Layout.BytesPerSample
			for _, p := range []goav1.FramePlane{f.Y, f.U, f.V} {
				for r := 0; r < p.Height; r++ {
					h.Write(p.Pix[r*p.Stride : r*p.Stride+p.Width*bps])
				}
			}
		}
	}
	pending = nil
	dec.Reset()
	return map[string]any{
		"frames":     frames,
		"msPerFrame": float64(best.Microseconds()) / 1000 / float64(max(frames, 1)),
		"md5":        hex.EncodeToString(h.Sum(nil)),
		"allocs":     float64(after.Mallocs-before.Mallocs) / float64(decoded),
		"allocBytes": float64(after.TotalAlloc-before.TotalAlloc) / float64(decoded),
		"gcs":        int(after.NumGC - before.NumGC),
	}
}
