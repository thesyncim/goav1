package prediction

import (
	"errors"
	"slices"
	"testing"
)

const (
	libaomIntraEdgeDeterministicSeed = 0xbaba
	libaomIntraEdgeIterations        = 128
)

func TestFilterIntraEdgeMatchesLibaomKnownVectors(t *testing.T) {
	tests := []struct {
		name     string
		bitDepth uint8
		strength uint8
		edge     []uint16
		want     []uint16
	}{
		{
			name:     "strength-zero",
			bitDepth: 8,
			strength: 0,
			edge:     []uint16{2, 5, 9, 17, 33},
			want:     []uint16{2, 5, 9, 17, 33},
		},
		{
			name:     "eight-bit-strength-one",
			bitDepth: 8,
			strength: 1,
			edge:     []uint16{10, 20, 40, 80, 160},
			want:     []uint16{10, 23, 45, 90, 140},
		},
		{
			name:     "ten-bit-strength-two",
			bitDepth: 10,
			strength: 2,
			edge:     []uint16{0, 1023, 512, 17, 999, 33},
			want:     []uint16{0, 544, 517, 479, 390, 335},
		},
		{
			name:     "twelve-bit-strength-three",
			bitDepth: 12,
			strength: 3,
			edge:     []uint16{0, 4095, 3000, 7, 2048},
			want:     []uint16{0, 1775, 2032, 2032, 1657},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := slices.Clone(tt.edge)
			scratch := make([]uint16, len(got))
			if err := FilterIntraEdge(got, scratch, tt.strength, tt.bitDepth); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, tt.want) {
				t.Fatalf("edge=%v want %v", got, tt.want)
			}
		})
	}
}

func TestFilterIntraEdgeMatchesLibaomOperationCorpus(t *testing.T) {
	for _, bitDepth := range []uint8{8, 10, 12} {
		hi := 1 << bitDepth
		rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
		for iter := 0; iter < libaomIntraEdgeIterations; iter++ {
			strength := uint8(rnd.pseudoUniform(4))
			size := 4*(rnd.pseudoUniform(128/4)+1) + 1
			got := make([]uint16, size)
			for i := range got {
				if bitDepth == 8 {
					got[i] = uint16(rnd.rand8())
				} else {
					got[i] = uint16(rnd.pseudoUniform(hi))
				}
			}
			want := slices.Clone(got)
			filterIntraEdgeLibaomReference(want, strength)

			scratch := make([]uint16, size)
			if err := FilterIntraEdge(got, scratch, strength, bitDepth); err != nil {
				t.Fatalf("bitDepth=%d iter=%d strength=%d size=%d: %v", bitDepth, iter, strength, size, err)
			}
			if !slices.Equal(got, want) {
				t.Fatalf("bitDepth=%d iter=%d strength=%d size=%d got=%v want=%v", bitDepth, iter, strength, size, got, want)
			}
		}
	}
}

func TestFilterIntraEdgeRejectsInvalidInputs(t *testing.T) {
	scratch := make([]uint16, intraEdgeMaxSize)
	tests := []struct {
		name     string
		edge     []uint16
		scratch  []uint16
		strength uint8
		bitDepth uint8
	}{
		{name: "empty", edge: nil, scratch: scratch, strength: 1, bitDepth: 8},
		{name: "too-large", edge: make([]uint16, intraEdgeMaxSize+1), scratch: make([]uint16, intraEdgeMaxSize+1), strength: 1, bitDepth: 8},
		{name: "invalid-strength", edge: []uint16{1, 2, 3}, scratch: scratch, strength: 4, bitDepth: 8},
		{name: "invalid-bitdepth", edge: []uint16{1, 2, 3}, scratch: scratch, strength: 1, bitDepth: 9},
		{name: "sample-out-of-range", edge: []uint16{1, 256, 3}, scratch: scratch, strength: 1, bitDepth: 8},
		{name: "short-scratch", edge: []uint16{1, 2, 3}, scratch: []uint16{0, 0}, strength: 1, bitDepth: 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := FilterIntraEdge(tt.edge, tt.scratch, tt.strength, tt.bitDepth); !errors.Is(err, ErrInvalidPrediction) {
				t.Fatalf("err=%v want %v", err, ErrInvalidPrediction)
			}
		})
	}
}

func TestFilterIntraEdgeStrengthZeroDoesNotRequireScratch(t *testing.T) {
	edge := []uint16{1, 2, 3, 4}
	if err := FilterIntraEdge(edge, nil, 0, 8); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(edge, []uint16{1, 2, 3, 4}) {
		t.Fatalf("edge changed: %v", edge)
	}
}

func TestFilterIntraEdgeAllocs(t *testing.T) {
	edge := make([]uint16, intraEdgeMaxSize)
	scratch := make([]uint16, intraEdgeMaxSize)
	for i := range edge {
		edge[i] = uint16((i * 17) & 0xfff)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := FilterIntraEdge(edge, scratch, 3, 12); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FilterIntraEdge allocated: %f", allocs)
	}
}

func FuzzFilterIntraEdge(f *testing.F) {
	f.Add([]byte{0, 16, 32, 64, 128, 255}, uint8(1), uint8(0))
	f.Add([]byte{255, 0, 127, 8, 240}, uint8(2), uint8(1))
	f.Add([]byte{1, 2, 3, 4}, uint8(3), uint8(2))

	f.Fuzz(func(t *testing.T, data []byte, rawStrength uint8, rawBitDepth uint8) {
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawBitDepth%uint8(len(bitDepths))]
		max := uint16((1 << bitDepth) - 1)
		size := len(data)%intraEdgeMaxSize + 1
		edge := make([]uint16, size)
		for i := range edge {
			if len(data) == 0 {
				continue
			}
			lo := uint16(data[i%len(data)])
			hi := uint16(data[(i+1)%len(data)])
			edge[i] = (lo | hi<<8) & max
		}
		before := slices.Clone(edge)
		strength := rawStrength % 4
		scratch := make([]uint16, size)
		if err := FilterIntraEdge(edge, scratch, strength, bitDepth); err != nil {
			t.Fatalf("FilterIntraEdge err=%v", err)
		}
		for i, sample := range edge {
			if sample > max {
				t.Fatalf("edge[%d]=%d exceeds max %d", i, sample, max)
			}
		}
		if strength == 0 && !slices.Equal(edge, before) {
			t.Fatalf("strength-zero edge=%v want %v", edge, before)
		}
	})
}

func BenchmarkFilterIntraEdge(b *testing.B) {
	edge := make([]uint16, intraEdgeMaxSize)
	scratch := make([]uint16, intraEdgeMaxSize)
	for i := range edge {
		edge[i] = uint16((i * 31) & 0xfff)
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = FilterIntraEdge(edge, scratch, 3, 12)
	}
}

func TestUpsampleIntraEdgeMatchesLibaomKnownVectors(t *testing.T) {
	tests := []struct {
		name     string
		bitDepth uint8
		edge     []uint16
		origin   int
		size     int
		want     []uint16
	}{
		{
			name:     "eight-bit",
			bitDepth: 8,
			edge:     []uint16{99, 99, 0, 10, 20, 40, 80, 160, 0, 0, 0},
			origin:   4,
			size:     4,
			want:     []uint16{10, 14, 20, 28, 40, 56, 80, 123, 160},
		},
		{
			name:     "twelve-bit-clipped",
			bitDepth: 12,
			edge:     []uint16{77, 77, 0, 0, 4095, 4095, 0, 0, 0, 0, 0},
			origin:   4,
			size:     4,
			want:     []uint16{0, 2048, 4095, 4095, 4095, 2048, 0, 0, 0},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := slices.Clone(tt.edge)
			scratch := make([]uint16, tt.size+3)
			if err := UpsampleIntraEdge(got, tt.origin, tt.size, scratch, tt.bitDepth); err != nil {
				t.Fatal(err)
			}
			window := got[tt.origin-2 : tt.origin+2*tt.size-1]
			if !slices.Equal(window, tt.want) {
				t.Fatalf("edge=%v want %v", window, tt.want)
			}
		})
	}
}

func TestUpsampleIntraEdgeMatchesLibaomOperationCorpus(t *testing.T) {
	const (
		offset  = 16
		bufSize = 2*64 + 32
	)
	for _, bitDepth := range []uint8{8, 10, 12} {
		hi := 1 << bitDepth
		rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
		for iter := 0; iter < libaomIntraEdgeIterations; iter++ {
			size := 4 * (rnd.pseudoUniform(4) + 1)
			got := make([]uint16, bufSize)
			pix := uint16(0)
			for i := 0; i < offset+size; i++ {
				if bitDepth == 8 {
					pix = uint16(rnd.rand8())
				} else {
					pix = uint16(rnd.pseudoUniform(hi))
				}
				got[i] = pix
			}
			for i := offset + size; i < len(got); i++ {
				got[i] = pix
			}
			want := slices.Clone(got)
			upsampleIntraEdgeLibaomReference(want, offset, size, bitDepth)

			scratch := make([]uint16, size+3)
			if err := UpsampleIntraEdge(got, offset, size, scratch, bitDepth); err != nil {
				t.Fatalf("bitDepth=%d iter=%d size=%d: %v", bitDepth, iter, size, err)
			}
			end := offset + (size-1)*2 + 1
			if !slices.Equal(got[offset-2:end], want[offset-2:end]) {
				t.Fatalf("bitDepth=%d iter=%d size=%d got=%v want=%v", bitDepth, iter, size, got[offset-2:end], want[offset-2:end])
			}
		}
	}
}

func TestUpsampleIntraEdgeRejectsInvalidInputs(t *testing.T) {
	edge := []uint16{0, 0, 1, 2, 3, 4, 0, 0, 0}
	scratch := make([]uint16, intraEdgeMaxUpsampleSize+3)
	tests := []struct {
		name     string
		edge     []uint16
		origin   int
		size     int
		scratch  []uint16
		bitDepth uint8
	}{
		{name: "zero-size", edge: edge, origin: 2, size: 0, scratch: scratch, bitDepth: 8},
		{name: "too-large", edge: make([]uint16, 64), origin: 2, size: intraEdgeMaxUpsampleSize + 1, scratch: make([]uint16, intraEdgeMaxUpsampleSize+4), bitDepth: 8},
		{name: "origin-before-negative-window", edge: edge, origin: 1, size: 4, scratch: scratch, bitDepth: 8},
		{name: "source-out-of-range", edge: []uint16{0, 0, 1}, origin: 2, size: 2, scratch: scratch, bitDepth: 8},
		{name: "write-out-of-range", edge: edge[:7], origin: 2, size: 4, scratch: scratch, bitDepth: 8},
		{name: "short-scratch", edge: edge, origin: 2, size: 4, scratch: []uint16{0, 0, 0, 0, 0, 0}, bitDepth: 8},
		{name: "invalid-bitdepth", edge: edge, origin: 2, size: 4, scratch: scratch, bitDepth: 9},
		{name: "sample-out-of-range", edge: []uint16{0, 0, 1, 256, 3, 4, 0, 0, 0}, origin: 2, size: 4, scratch: scratch, bitDepth: 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := UpsampleIntraEdge(tt.edge, tt.origin, tt.size, tt.scratch, tt.bitDepth); !errors.Is(err, ErrInvalidPrediction) {
				t.Fatalf("err=%v want %v", err, ErrInvalidPrediction)
			}
		})
	}
}

func TestUpsampleIntraEdgeAllocs(t *testing.T) {
	edge := make([]uint16, 64)
	const origin = 16
	const size = 16
	for i := origin - 1; i < origin+size; i++ {
		edge[i] = uint16((i * 29) & 0x3ff)
	}
	for i := origin + size; i < len(edge); i++ {
		edge[i] = edge[origin+size-1]
	}
	scratch := make([]uint16, size+3)

	allocs := testing.AllocsPerRun(1000, func() {
		if err := UpsampleIntraEdge(edge, origin, size, scratch, 10); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("UpsampleIntraEdge allocated: %f", allocs)
	}
}

func FuzzUpsampleIntraEdge(f *testing.F) {
	f.Add([]byte{0, 10, 20, 40, 80, 160, 160}, uint8(0), uint8(0))
	f.Add([]byte{0xff, 0, 0xff, 0, 0x80, 0x40}, uint8(1), uint8(2))

	f.Fuzz(func(t *testing.T, data []byte, rawSize uint8, rawBitDepth uint8) {
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawBitDepth%uint8(len(bitDepths))]
		max := uint16((1 << bitDepth) - 1)
		size := 4 * (int(rawSize)%4 + 1)
		origin := 16
		edge := make([]uint16, origin+2*size-1)
		for i := origin - 1; i < origin+size; i++ {
			if len(data) == 0 {
				continue
			}
			lo := uint16(data[i%len(data)])
			hi := uint16(data[(i+1)%len(data)])
			edge[i] = (lo | hi<<8) & max
		}
		for i := origin + size; i < len(edge); i++ {
			edge[i] = edge[origin+size-1]
		}
		scratch := make([]uint16, size+3)
		if err := UpsampleIntraEdge(edge, origin, size, scratch, bitDepth); err != nil {
			t.Fatalf("UpsampleIntraEdge err=%v", err)
		}
		for i, sample := range edge[origin-2:] {
			if sample > max {
				t.Fatalf("edge[%d]=%d exceeds max %d", origin-2+i, sample, max)
			}
		}
	})
}

func BenchmarkUpsampleIntraEdge(b *testing.B) {
	edge := make([]uint16, 64)
	const origin = 16
	const size = 16
	for i := origin - 1; i < origin+size; i++ {
		edge[i] = uint16((i * 31) & 0xfff)
	}
	for i := origin + size; i < len(edge); i++ {
		edge[i] = edge[origin+size-1]
	}
	scratch := make([]uint16, size+3)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = UpsampleIntraEdge(edge, origin, size, scratch, 12)
	}
}

func filterIntraEdgeLibaomReference(edge []uint16, strength uint8) {
	if strength == 0 {
		return
	}
	scratch := slices.Clone(edge)
	kernel := intraEdgeFilterKernels[strength-1]
	for i := 1; i < len(edge); i++ {
		sum := 0
		for j := 0; j < intraEdgeTaps; j++ {
			k := i - 2 + j
			if k < 0 {
				k = 0
			} else if k > len(edge)-1 {
				k = len(edge) - 1
			}
			sum += int(scratch[k]) * kernel[j]
		}
		edge[i] = uint16((sum + 8) >> 4)
	}
}

func upsampleIntraEdgeLibaomReference(edge []uint16, origin int, size int, bitDepth uint8) {
	max := int((1 << bitDepth) - 1)
	scratch := make([]uint16, size+3)
	scratch[0] = edge[origin-1]
	scratch[1] = edge[origin-1]
	copy(scratch[2:2+size], edge[origin:origin+size])
	scratch[size+2] = edge[origin+size-1]

	edge[origin-2] = scratch[0]
	for i := 0; i < size; i++ {
		sum := -int(scratch[i]) + 9*int(scratch[i+1]) + 9*int(scratch[i+2]) - int(scratch[i+3])
		sample := (sum + 8) >> 4
		if sample < 0 {
			sample = 0
		} else if sample > max {
			sample = max
		}
		edge[origin+2*i-1] = uint16(sample)
		edge[origin+2*i] = scratch[i+2]
	}
}

type libaomIntraEdgeRandom struct {
	state uint32
}

func newLibaomIntraEdgeRandom(seed uint32) *libaomIntraEdgeRandom {
	return &libaomIntraEdgeRandom{state: seed}
}

func (r *libaomIntraEdgeRandom) generate(randomRange uint32) uint32 {
	r.state = (1103515245*r.state + 12345) & ((1 << 31) - 1)
	return r.state % randomRange
}

func (r *libaomIntraEdgeRandom) rand8() uint8 {
	return uint8((r.generate(1<<31) >> 23) & 0xff)
}

func (r *libaomIntraEdgeRandom) pseudoUniform(randomRange int) int {
	return int(r.generate(uint32(randomRange)))
}
