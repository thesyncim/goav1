// Package pattern generates the deterministic synthetic scene the encoder
// benchmark and the verification run encode. The native verify command and the
// wasm encoder share it, so both encode identical frames.
package pattern

import goav1 "github.com/thesyncim/goav1"

// Fill writes frame t of the scene into f: a XOR texture panning with t, a
// bright disc sweeping across the middle, and slowly shifting chroma ramps.
// f must have even Width and Height and planes of at least the matching size.
func Fill(f goav1.I420Frame, t int) {
	w, h := f.Width, f.Height
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := (x + 2*t) ^ (y + t)
			if (x-w/2-3*t%200)*(x-w/2-3*t%200)+(y-h/2)*(y-h/2) < 60*60 {
				v = 230
			}
			f.Y[y*f.YStride+x] = byte(v)
		}
	}
	for y := 0; y < h/2; y++ {
		for x := 0; x < w/2; x++ {
			f.U[y*f.ChromaStride+x] = byte(128 + (x+t)%32)
			f.V[y*f.ChromaStride+x] = byte(128 - (y+t)%32)
		}
	}
}
