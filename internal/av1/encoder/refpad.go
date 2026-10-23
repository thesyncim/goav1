package encoder

import "github.com/thesyncim/goav1/internal/av1/frame"

// refpad.go keeps edge-extended copies of the reference reconstructions, the
// encoder side of SVT-AV1's padded reference pictures (svt_aom_generate_padding
// on every reference picture, read directly by svt_aom_inter_pu_prediction
// and the sub-pel search with no per-block clamping). Replicating the edge
// samples outward is exactly the decoder's reference clamping, so a window
// that stays inside the border reads the same samples the clamped paths
// build; windows reaching past it keep those paths. The reconstructions
// themselves (Recon(), the decoder-equal output) are not changed.

const (
	// refPadBorderLuma covers SVT's light-PD1 search and prediction reach
	// beyond the picture (PAD_VALUE-scale borders); chroma uses half.
	refPadBorderLuma   = 80
	refPadBorderChroma = refPadBorderLuma / 2
	refPadCacheSize    = 4
)

// refPadPlane is one padded plane: buf holds (w+2*border) x (h+2*border)
// samples at stride, the source plane's (0,0) at (border, border).
type refPadPlane struct {
	buf          []byte
	stride       int
	border, w, h int
	src          *byte
}

func (p *refPadPlane) build(src []byte, srcStride, w, h, border int) {
	p.stride = w + 2*border
	need := p.stride * (h + 2*border)
	if cap(p.buf) < need {
		p.buf = make([]byte, need)
	}
	p.buf = p.buf[:need]
	p.border, p.w, p.h = border, w, h
	p.src = &src[0]
	for r := 0; r < h; r++ {
		row := p.buf[(r+border)*p.stride : (r+border+1)*p.stride]
		s := src[r*srcStride : r*srcStride+w]
		frame.FillBytes(row[:border], s[0])
		copy(row[border:border+w], s)
		frame.FillBytes(row[border+w:], s[w-1])
	}
	top := p.buf[border*p.stride : (border+1)*p.stride]
	for r := 0; r < border; r++ {
		copy(p.buf[r*p.stride:(r+1)*p.stride], top)
	}
	bottom := p.buf[(border+h-1)*p.stride : (border+h)*p.stride]
	for r := border + h; r < h+2*border; r++ {
		copy(p.buf[r*p.stride:(r+1)*p.stride], bottom)
	}
}

// plane is the padded plane as a frame.Plane whose origin is the padded
// top-left: a source position (x, y) is (x+border, y+border).
func (p *refPadPlane) plane() frame.Plane {
	return frame.Plane{Pix: p.buf, Stride: p.stride, Width: p.w + 2*p.border, Height: p.h + 2*p.border}
}

// refPadFrame is one padded reference picture.
type refPadFrame struct {
	planes [3]refPadPlane
	key    *byte
}

// refPads is the set of padded references a tile coder reads.
type refPads struct {
	frames []*refPadFrame
}

// lookup returns the padded copy of refPlane (w x h), nil without one.
func (rp *refPads) lookup(refPlane []byte, w, h int) *refPadPlane {
	if rp == nil || len(refPlane) == 0 {
		return nil
	}
	ptr := &refPlane[0]
	for _, f := range rp.frames {
		for i := range f.planes {
			p := &f.planes[i]
			if p.src == ptr && p.w == w && p.h == h {
				return p
			}
		}
	}
	return nil
}

// refPadCache holds padded copies keyed by the reconstruction buffer they
// mirror. A buffer is invalidated whenever an encode writes it, so an entry
// is only reused while its reconstruction is unchanged.
type refPadCache struct {
	entries [refPadCacheSize]refPadFrame
	next    int
	set     refPads
	setBuf  [2]*refPadFrame
}

func (c *refPadCache) invalidate(f *SourceFrame420) {
	if f == nil || len(f.Y) == 0 {
		return
	}
	for i := range c.entries {
		if c.entries[i].key == &f.Y[0] {
			c.entries[i].key = nil
		}
	}
}

func (c *refPadCache) invalidateAll() {
	for i := range c.entries {
		c.entries[i].key = nil
	}
}

// get returns the padded copy of f, building it when absent.
func (c *refPadCache) get(f SourceFrame420, cw, ch int) *refPadFrame {
	key := &f.Y[0]
	c.reserve(f.Width, f.Height, cw, ch)
	for i := range c.entries {
		e := &c.entries[i]
		if e.key == key && e.planes[0].w == f.Width && e.planes[0].h == f.Height {
			return e
		}
	}
	e := &c.entries[c.next]
	c.next = (c.next + 1) % refPadCacheSize
	e.planes[0].build(f.Y, f.YStride, f.Width, f.Height, refPadBorderLuma)
	e.planes[1].build(f.U, f.ChromaStride, cw, ch, refPadBorderChroma)
	e.planes[2].build(f.V, f.ChromaStride, cw, ch, refPadBorderChroma)
	e.key = key
	return e
}

// padsFor returns the padded set for the frame's references (golden may be
// nil). The returned set is valid until the next call.
func (c *refPadCache) padsFor(ref SourceFrame420, golden *SourceFrame420, cw, ch int) *refPads {
	if len(ref.Y) == 0 || ref.Width <= 0 || ref.Height <= 0 || cw <= 0 || ch <= 0 {
		return nil
	}
	c.setBuf[0] = c.get(ref, cw, ch)
	c.set.frames = c.setBuf[:1]
	if golden != nil && len(golden.Y) != 0 && golden.Width == ref.Width && golden.Height == ref.Height && &golden.Y[0] != &ref.Y[0] {
		c.setBuf[1] = c.get(*golden, cw, ch)
		c.set.frames = c.setBuf[:2]
	}
	return &c.set
}

// dropRefPads forgets every padded copy and detaches them from the tile
// coders, for the encode paths that write reference buffers without them.
func (e *VideoEncoder) dropRefPads() {
	e.refPad.invalidateAll()
	e.pc.st.refPads = nil
	for t := range e.tilePCs {
		e.tilePCs[t].st.refPads = nil
	}
}

// reserve sizes every cache entry for a w x h picture at once, so the
// steady state allocates nothing however the entries rotate.
func (c *refPadCache) reserve(w, h, cw, ch int) {
	lumaNeed := (w + 2*refPadBorderLuma) * (h + 2*refPadBorderLuma)
	chromaNeed := (cw + 2*refPadBorderChroma) * (ch + 2*refPadBorderChroma)
	for i := range c.entries {
		p := &c.entries[i].planes
		if cap(p[0].buf) < lumaNeed {
			p[0].buf = make([]byte, lumaNeed)
		}
		for j := 1; j < 3; j++ {
			if cap(p[j].buf) < chromaNeed {
				p[j].buf = make([]byte, chromaNeed)
			}
		}
	}
}
