package cdef

import "testing"

const cdefDeterministicSeed = 0xbaba

func TestFindDirectionKnownPatterns(t *testing.T) {
	const stride = 8
	const size = stride * 8
	tests := []struct {
		name    string
		fill    func([]uint16)
		wantDir int
		wantVar int32
	}{
		{
			name: "constant",
			fill: func(img []uint16) {
				for i := range img {
					img[i] = 128
				}
			},
			wantDir: 0,
			wantVar: 0,
		},
		{
			name: "horizontal",
			fill: func(img []uint16) {
				for row := range 8 {
					for col := range 8 {
						img[row*stride+col] = uint16(row * 32)
					}
				}
			},
			wantDir: 2,
			wantVar: 282240,
		},
		{
			name: "vertical",
			fill: func(img []uint16) {
				for row := range 8 {
					for col := range 8 {
						img[row*stride+col] = uint16(col * 32)
					}
				}
			},
			wantDir: 6,
			wantVar: 282240,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			img := make([]uint16, size)
			tt.fill(img)
			gotDir, gotVar, err := FindDirection(img, stride, 0)
			if err != nil {
				t.Fatal(err)
			}
			if gotDir != tt.wantDir || gotVar != tt.wantVar {
				t.Fatalf("dir,var=%d,%d want %d,%d", gotDir, gotVar, tt.wantDir, tt.wantVar)
			}
		})
	}
}

func TestFindDirectionMatchesLibaomCorpus(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed)
	for depth := 8; depth <= 12; depth += 2 {
		max := uint16((1 << depth) - 1)
		coeffShift := depth - 8
		for iter := range 128 {
			stride := 8 + rnd.pseudoUniform(9)
			img := make([]uint16, stride*8)
			for i := range img {
				img[i] = uint16(rnd.pseudoUniform(int(max) + 1))
			}
			wantDir, wantVar := findDirectionLibaomReference(img, stride, coeffShift)
			gotDir, gotVar, err := FindDirection(img, stride, coeffShift)
			if err != nil {
				t.Fatalf("depth=%d iter=%d: %v", depth, iter, err)
			}
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("depth=%d iter=%d dir,var=%d,%d want %d,%d", depth, iter, gotDir, gotVar, wantDir, wantVar)
			}
		}
	}
}

func TestFindDirectionDualMatchesSingle(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed)
	stride := 16
	img := make([]uint16, stride*8)
	for i := range img {
		img[i] = uint16(rnd.pseudoUniform(1 << 10))
	}
	wantDir1, wantVar1, err := FindDirection(img, stride, 2)
	if err != nil {
		t.Fatal(err)
	}
	wantDir2, wantVar2, err := FindDirection(img[8:], stride, 2)
	if err != nil {
		t.Fatal(err)
	}
	gotDir1, gotVar1, gotDir2, gotVar2, err := FindDirectionDual(img, img[8:], stride, 2)
	if err != nil {
		t.Fatal(err)
	}
	if gotDir1 != wantDir1 || gotVar1 != wantVar1 || gotDir2 != wantDir2 || gotVar2 != wantVar2 {
		t.Fatalf("dual=(%d,%d),(%d,%d) want (%d,%d),(%d,%d)", gotDir1, gotVar1, gotDir2, gotVar2, wantDir1, wantVar1, wantDir2, wantVar2)
	}
}

func TestFindDirectionRejectsInvalidInputs(t *testing.T) {
	img := make([]uint16, 64)
	tests := []struct {
		name       string
		img        []uint16
		stride     int
		coeffShift int
	}{
		{name: "short", img: img[:63], stride: 8},
		{name: "small-stride", img: img, stride: 7},
		{name: "negative-shift", img: img, stride: 8, coeffShift: -1},
		{name: "large-shift", img: img, stride: 8, coeffShift: 5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, _, err := FindDirection(tt.img, tt.stride, tt.coeffShift); err != ErrInvalidCDEF {
				t.Fatalf("err=%v want %v", err, ErrInvalidCDEF)
			}
		})
	}
}

func TestFindDirectionAllocs(t *testing.T) {
	img := make([]uint16, 64)
	for i := range img {
		img[i] = uint16((i * 37) & 0xfff)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if _, _, err := FindDirection(img, 8, 4); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FindDirection allocated: %f", allocs)
	}
}

func FuzzFindDirection(f *testing.F) {
	f.Add([]byte{0, 16, 32, 64, 128, 255}, uint8(0), uint8(0))
	f.Add([]byte{255, 0, 127, 32, 64}, uint8(4), uint8(2))

	f.Fuzz(func(t *testing.T, data []byte, rawStride uint8, rawCoeffShift uint8) {
		stride := 8 + int(rawStride%8)
		coeffShift := int(rawCoeffShift % 5)
		img := make([]uint16, stride*8)
		max := uint16((1 << (coeffShift + 8)) - 1)
		for i := range img {
			if len(data) == 0 {
				continue
			}
			lo := uint16(data[i%len(data)])
			hi := uint16(data[(i+1)%len(data)])
			img[i] = (lo | hi<<8) & max
		}
		dir, variance, err := FindDirection(img, stride, coeffShift)
		if err != nil {
			t.Fatalf("FindDirection err=%v", err)
		}
		if dir < 0 || dir > 7 {
			t.Fatalf("dir=%d out of range", dir)
		}
		if variance < 0 {
			t.Fatalf("variance=%d want non-negative", variance)
		}
	})
}

func BenchmarkFindDirection(b *testing.B) {
	img := make([]uint16, 64)
	for i := range img {
		img[i] = uint16((i * 41) & 0xfff)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _, _ = FindDirection(img, 8, 4)
	}
}

func BenchmarkFindDirection8(b *testing.B) {
	img := make([]uint16, 64)
	for i := range img {
		img[i] = uint16((i * 41) & 0xff)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _, _ = FindDirection(img, 8, 0)
	}
}

func BenchmarkFindDirectionDual(b *testing.B) {
	const stride = 16
	img := make([]uint16, stride*8)
	for i := range img {
		img[i] = uint16((i * 41) & 0xfff)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _, _ = FindDirectionDual(img, img[8:], stride, 4)
	}
}

func BenchmarkFindDirectionDual8(b *testing.B) {
	const stride = 16
	img := make([]uint16, stride*8)
	for i := range img {
		img[i] = uint16((i * 41) & 0xff)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _, _ = FindDirectionDual(img, img[8:], stride, 0)
	}
}

func findDirectionLibaomReference(img []uint16, stride int, coeffShift int) (int, int32) {
	var cost [8]int32
	var partial [8][15]int
	divTable := [...]int{0, 840, 420, 280, 210, 168, 140, 120, 105}
	for i := range 8 {
		for j := range 8 {
			x := int(img[i*stride+j]>>coeffShift) - 128
			partial[0][i+j] += x
			partial[1][i+j/2] += x
			partial[2][i] += x
			partial[3][3+i-j/2] += x
			partial[4][7+i-j] += x
			partial[5][3-i/2+j] += x
			partial[6][j] += x
			partial[7][i/2+j] += x
		}
	}
	for i := range 8 {
		cost[2] += int32(partial[2][i] * partial[2][i])
		cost[6] += int32(partial[6][i] * partial[6][i])
	}
	cost[2] *= int32(divTable[8])
	cost[6] *= int32(divTable[8])
	for i := range 7 {
		cost[0] += int32((partial[0][i]*partial[0][i] + partial[0][14-i]*partial[0][14-i]) * divTable[i+1])
		cost[4] += int32((partial[4][i]*partial[4][i] + partial[4][14-i]*partial[4][14-i]) * divTable[i+1])
	}
	cost[0] += int32(partial[0][7] * partial[0][7] * divTable[8])
	cost[4] += int32(partial[4][7] * partial[4][7] * divTable[8])
	for i := 1; i < 8; i += 2 {
		for j := range 5 {
			cost[i] += int32(partial[i][3+j] * partial[i][3+j])
		}
		cost[i] *= int32(divTable[8])
		for j := range 3 {
			cost[i] += int32((partial[i][j]*partial[i][j] + partial[i][10-j]*partial[i][10-j]) * divTable[2*j+2])
		}
	}
	bestCost := int32(0)
	bestDir := 0
	for i := range 8 {
		if cost[i] > bestCost {
			bestCost = cost[i]
			bestDir = i
		}
	}
	return bestDir, (bestCost - cost[(bestDir+4)&7]) >> 10
}

type cdefRandom struct {
	state uint32
}

func newCDEFRandom(seed uint32) *cdefRandom {
	return &cdefRandom{state: seed}
}

func (r *cdefRandom) generate(randomRange uint32) uint32 {
	r.state = (1103515245*r.state + 12345) & ((1 << 31) - 1)
	return r.state % randomRange
}

func (r *cdefRandom) pseudoUniform(randomRange int) int {
	return int(r.generate(uint32(randomRange)))
}
