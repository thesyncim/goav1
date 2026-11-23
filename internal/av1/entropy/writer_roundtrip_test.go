// Writer/Reader round trip over randomized symbol streams: every write
// primitive, including the coefficient base/BR/sign/Golomb run, is decoded back
// with the same initial CDF image and must reproduce the symbols and the
// adapted CDFs.
package entropy

import (
	"math/rand"
	"sort"
	"strconv"
	"testing"
)

const (
	roundTripRowBinary = iota
	roundTripRowCDF3
	roundTripRowCDF4
	roundTripRowCDF5
	roundTripRowCDF7
	roundTripRowCDF8
	roundTripRowCDF16
	roundTripRows
)

var roundTripRowSymbols = [roundTripRows]int{2, 3, 4, 5, 7, 8, 16}

type roundTripCDFs struct {
	Rows [roundTripRows]CDF
}

type roundTripFixedCDF struct {
	symbols int
	values  []uint16
}

type roundTripRecord struct {
	bytes []byte
	cdfs  roundTripCDFs
	err   error
}

type roundTripOpKind uint8

const (
	roundTripOpBoolQ15 roundTripOpKind = iota
	roundTripOpBit
	roundTripOpLiteral
	roundTripOpFixedSymbol
	roundTripOpAdaptiveSymbol
	roundTripOpCDF
	roundTripOpBinaryCDF
	roundTripOpCDF3
	roundTripOpCDF4
	roundTripOpCDF4Zero
	roundTripOpCDF5
	roundTripOpCDF7
	roundTripOpCoeffRun
)

type roundTripOp struct {
	kind  roundTripOpKind
	row   uint8
	sym   uint8
	prob  uint16
	nbits uint8
	value uint32
	level uint16
	sign  uint8
	dc    bool
}

func roundTripICDFFromWeights(t *testing.T, weights []int) []uint16 {
	t.Helper()
	total := 0
	for _, weight := range weights {
		if weight <= 0 {
			t.Fatalf("invalid fixed CDF weight %d", weight)
		}
		total += weight
	}
	cumulative := make([]uint16, len(weights)-1)
	sum := 0
	for i := range cumulative {
		sum += weights[i]
		v := (sum * CDFProbTop) / total
		if v <= 0 || v >= CDFProbTop {
			t.Fatalf("invalid fixed CDF cumulative %d from weights %v", v, weights)
		}
		cumulative[i] = uint16(v)
	}
	icdf := make([]uint16, len(weights)+1)
	if err := InitCDF(icdf, cumulative); err != nil {
		t.Fatalf("InitCDF fixed %v: %v", weights, err)
	}
	return icdf
}

func roundTripFixedCDFs(t *testing.T) []roundTripFixedCDF {
	t.Helper()
	weights := [][]int{
		{1, 5},
		{3, 1, 4, 1},
		{2, 7, 1, 8, 2, 8, 1, 8},
		{1, 1, 2, 3, 5, 8, 13, 21, 3, 2, 1, 4, 6, 7, 9, 11},
	}
	out := make([]roundTripFixedCDF, len(weights))
	for i, row := range weights {
		out[i] = roundTripFixedCDF{
			symbols: len(row),
			values:  roundTripICDFFromWeights(t, row),
		}
	}
	return out
}

func roundTripSeedRow(t *testing.T, rng *rand.Rand, cdf *CDF, symbols, shape int) {
	t.Helper()
	if symbols < 2 || symbols > MaxSymbols {
		t.Fatalf("invalid symbols=%d", symbols)
	}
	cumulative := make([]uint16, symbols-1)
	switch shape % 4 {
	case 0:
		if err := cdf.InitUniform(symbols); err != nil {
			t.Fatalf("InitUniform(%d): %v", symbols, err)
		}
	case 1:
		v := 0
		for i := range cumulative {
			v += 1 + rng.Intn(3)
			cumulative[i] = uint16(v)
		}
		if err := cdf.Init(cumulative); err != nil {
			t.Fatalf("tail-heavy Init(%d): %v", symbols, err)
		}
	case 2:
		v := CDFProbTop
		for i := len(cumulative) - 1; i >= 0; i-- {
			v -= 1 + rng.Intn(3)
			cumulative[i] = uint16(v)
		}
		if err := cdf.Init(cumulative); err != nil {
			t.Fatalf("head-heavy Init(%d): %v", symbols, err)
		}
	default:
		vals := make([]int, len(cumulative))
		for i := range vals {
			for {
				v := 1 + rng.Intn(CDFProbTop-1)
				used := false
				for j := 0; j < i; j++ {
					if vals[j] == v {
						used = true
						break
					}
				}
				if !used {
					vals[i] = v
					break
				}
			}
		}
		sort.Ints(vals)
		for i, v := range vals {
			cumulative[i] = uint16(v)
		}
		if err := cdf.Init(cumulative); err != nil {
			t.Fatalf("random Init(%d): %v", symbols, err)
		}
	}
	warm := rng.Intn(48)
	for i := 0; i < warm; i++ {
		var sym int
		switch shape % 4 {
		case 1:
			if rng.Intn(8) == 0 {
				sym = rng.Intn(symbols)
			} else {
				sym = symbols - 1
			}
		case 2:
			if rng.Intn(8) == 0 {
				sym = rng.Intn(symbols)
			} else {
				sym = 0
			}
		default:
			sym = rng.Intn(symbols)
		}
		if err := cdf.Update(sym); err != nil {
			t.Fatalf("warm Update(%d/%d): %v", sym, symbols, err)
		}
	}
}

func roundTripSeedCDFs(t *testing.T, rng *rand.Rand, shape int) roundTripCDFs {
	t.Helper()
	var cdfs roundTripCDFs
	for row, symbols := range roundTripRowSymbols {
		roundTripSeedRow(t, rng, &cdfs.Rows[row], symbols, shape+row)
	}
	return cdfs
}

func roundTripRandomOp(rng *rand.Rand, fixed []roundTripFixedCDF) roundTripOp {
	switch rng.Intn(13) {
	case 0:
		return roundTripOp{
			kind: roundTripOpBoolQ15,
			sym:  uint8(rng.Intn(2)),
			prob: uint16(1 + rng.Intn(CDFProbTop-1)),
		}
	case 1:
		return roundTripOp{kind: roundTripOpBit, sym: uint8(rng.Intn(2))}
	case 2:
		n := uint8(1 + rng.Intn(24))
		return roundTripOp{
			kind:  roundTripOpLiteral,
			nbits: n,
			value: rng.Uint32() & ((uint32(1) << n) - 1),
		}
	case 3:
		row := uint8(rng.Intn(len(fixed)))
		return roundTripOp{
			kind: roundTripOpFixedSymbol,
			row:  row,
			sym:  uint8(rng.Intn(fixed[row].symbols)),
		}
	case 4:
		row := uint8(roundTripRowCDF8 + rng.Intn(2))
		return roundTripOp{
			kind: roundTripOpAdaptiveSymbol,
			row:  row,
			sym:  uint8(rng.Intn(roundTripRowSymbols[row])),
		}
	case 5:
		row := uint8(rng.Intn(roundTripRows))
		return roundTripOp{
			kind: roundTripOpCDF,
			row:  row,
			sym:  uint8(rng.Intn(roundTripRowSymbols[row])),
		}
	case 6:
		return roundTripOp{kind: roundTripOpBinaryCDF, sym: uint8(rng.Intn(2))}
	case 7:
		return roundTripOp{kind: roundTripOpCDF3, sym: uint8(rng.Intn(3))}
	case 8:
		if rng.Intn(4) == 0 {
			return roundTripOp{kind: roundTripOpCDF4Zero}
		}
		return roundTripOp{kind: roundTripOpCDF4, sym: uint8(rng.Intn(4))}
	case 9:
		return roundTripOp{kind: roundTripOpCDF5, sym: uint8(rng.Intn(5))}
	case 10:
		return roundTripOp{kind: roundTripOpCDF7, sym: uint8(rng.Intn(7))}
	default:
		level := 1 + rng.Intn(40)
		if rng.Intn(5) == 0 {
			level = 15 + rng.Intn(64)
		}
		return roundTripOp{
			kind:  roundTripOpCoeffRun,
			level: uint16(level),
			sign:  uint8(rng.Intn(2)),
			dc:    rng.Intn(3) == 0,
		}
	}
}

func roundTripBuildOps(rng *rand.Rand, fixed []roundTripFixedCDF, n int) []roundTripOp {
	ops := make([]roundTripOp, n)
	for i := range ops {
		ops[i] = roundTripRandomOp(rng, fixed)
	}
	return ops
}

func roundTripWriteGolomb(w *Writer, level int) {
	x := level + 1
	length := 0
	for i := x; i != 0; i >>= 1 {
		length++
	}
	for range length - 1 {
		w.WriteBit(0)
	}
	for i := length - 1; i >= 0; i-- {
		w.WriteBit((x >> uint(i)) & 1)
	}
}

const (
	roundTripNumBaseLevels  = 2
	roundTripBaseRange      = 12
	roundTripBRCDFSize      = 4
	roundTripMaxBaseBRRange = 15
)

func roundTripWriteBR(w *Writer, cdf *CDF, level int) {
	baseRange := level - 1 - roundTripNumBaseLevels
	for idx := 0; idx < roundTripBaseRange; idx += roundTripBRCDFSize - 1 {
		k := min(baseRange-idx, roundTripBRCDFSize-1)
		w.WriteCDF4(cdf, k)
		if k < roundTripBRCDFSize-1 {
			break
		}
	}
}

func roundTripWriteCoeffRun(w *Writer, cdfs *roundTripCDFs, op roundTripOp) {
	level := int(op.level)
	baseSym := min(level, 3) - 1
	w.WriteCDF3(&cdfs.Rows[roundTripRowCDF3], baseSym)
	if level > roundTripNumBaseLevels {
		roundTripWriteBR(w, &cdfs.Rows[roundTripRowCDF4], level)
	}
	if op.dc {
		w.WriteBinaryCDFTrusted(&cdfs.Rows[roundTripRowBinary], int(op.sign))
	} else {
		w.WriteBit(int(op.sign))
	}
	if level >= roundTripMaxBaseBRRange {
		roundTripWriteGolomb(w, level-roundTripMaxBaseBRRange)
	}
}

func roundTripApplyOp(w *Writer, cdfs *roundTripCDFs, fixed []roundTripFixedCDF, op roundTripOp) {
	switch op.kind {
	case roundTripOpBoolQ15:
		w.WriteBoolQ15(int(op.sym), uint32(op.prob))
	case roundTripOpBit:
		w.WriteBit(int(op.sym))
	case roundTripOpLiteral:
		w.WriteLiteral(op.value, int(op.nbits))
	case roundTripOpFixedSymbol:
		row := fixed[op.row]
		w.WriteSymbol(int(op.sym), row.values, row.symbols)
	case roundTripOpAdaptiveSymbol:
		row := &cdfs.Rows[op.row]
		w.WriteSymbolAdaptive(int(op.sym), row.Values())
	case roundTripOpCDF:
		w.WriteCDF(&cdfs.Rows[op.row], int(op.sym))
	case roundTripOpBinaryCDF:
		w.WriteBinaryCDFTrusted(&cdfs.Rows[roundTripRowBinary], int(op.sym))
	case roundTripOpCDF3:
		w.WriteCDF3(&cdfs.Rows[roundTripRowCDF3], int(op.sym))
	case roundTripOpCDF4:
		w.WriteCDF4(&cdfs.Rows[roundTripRowCDF4], int(op.sym))
	case roundTripOpCDF4Zero:
		w.WriteCDF4Zero(&cdfs.Rows[roundTripRowCDF4])
	case roundTripOpCDF5:
		w.WriteCDF5(&cdfs.Rows[roundTripRowCDF5], int(op.sym))
	case roundTripOpCDF7:
		w.WriteCDF7(&cdfs.Rows[roundTripRowCDF7], int(op.sym))
	case roundTripOpCoeffRun:
		roundTripWriteCoeffRun(w, cdfs, op)
	}
}

func roundTripEncodeStream(ops []roundTripOp, initial roundTripCDFs, fixed []roundTripFixedCDF, dstCap int) roundTripRecord {
	cdfs := initial
	w := NewWriter(make([]byte, 0, dstCap))
	for _, op := range ops {
		roundTripApplyOp(&w, &cdfs, fixed, op)
	}
	buf, err := w.Finish()
	return roundTripRecord{bytes: buf, cdfs: cdfs, err: err}
}

func roundTripReadGolomb(t *testing.T, r *Reader, opIndex int) int {
	t.Helper()
	zeros := 0
	for {
		bit, err := r.ReadBit()
		if err != nil {
			t.Fatalf("op=%d golomb prefix: %v", opIndex, err)
		}
		if bit != 0 {
			break
		}
		zeros++
		if zeros > 32 {
			t.Fatalf("op=%d golomb prefix too long", opIndex)
		}
	}
	value := 1
	for i := 0; i < zeros; i++ {
		bit, err := r.ReadBit()
		if err != nil {
			t.Fatalf("op=%d golomb suffix[%d]: %v", opIndex, i, err)
		}
		value = (value << 1) | int(bit)
	}
	return value - 1
}

func roundTripReadSymbol(t *testing.T, r *Reader, cdf *CDF, want int, opIndex int, label string) {
	t.Helper()
	got, err := r.ReadSymbol(cdf.Values(), cdf.Symbols())
	if err != nil {
		t.Fatalf("op=%d %s ReadSymbol: %v", opIndex, label, err)
	}
	if got != want {
		t.Fatalf("op=%d %s symbol=%d want %d", opIndex, label, got, want)
	}
}

func roundTripReadBR(t *testing.T, r *Reader, cdf *CDF, level, opIndex int) {
	t.Helper()
	baseRange := level - 1 - roundTripNumBaseLevels
	for idx := 0; idx < roundTripBaseRange; idx += roundTripBRCDFSize - 1 {
		want := min(baseRange-idx, roundTripBRCDFSize-1)
		roundTripReadSymbol(t, r, cdf, want, opIndex, "coeff-br")
		if want < roundTripBRCDFSize-1 {
			break
		}
	}
}

func roundTripDecodeCoeffRun(t *testing.T, r *Reader, cdfs *roundTripCDFs, op roundTripOp, opIndex int) {
	t.Helper()
	level := int(op.level)
	roundTripReadSymbol(t, r, &cdfs.Rows[roundTripRowCDF3], min(level, 3)-1, opIndex, "coeff-base-eob")
	if level > roundTripNumBaseLevels {
		roundTripReadBR(t, r, &cdfs.Rows[roundTripRowCDF4], level, opIndex)
	}
	if op.dc {
		roundTripReadSymbol(t, r, &cdfs.Rows[roundTripRowBinary], int(op.sign), opIndex, "coeff-dc-sign")
	} else {
		got, err := r.ReadBit()
		if err != nil {
			t.Fatalf("op=%d coeff-sign ReadBit: %v", opIndex, err)
		}
		if got != op.sign {
			t.Fatalf("op=%d coeff-sign=%d want %d", opIndex, got, op.sign)
		}
	}
	if level >= roundTripMaxBaseBRRange {
		got := roundTripReadGolomb(t, r, opIndex)
		want := level - roundTripMaxBaseBRRange
		if got != want {
			t.Fatalf("op=%d coeff-golomb=%d want %d", opIndex, got, want)
		}
	}
}

func roundTripDecodeOp(t *testing.T, r *Reader, cdfs *roundTripCDFs, fixed []roundTripFixedCDF, op roundTripOp, opIndex int) {
	t.Helper()
	switch op.kind {
	case roundTripOpBoolQ15:
		got, err := r.ReadBoolQ15(op.prob)
		if err != nil {
			t.Fatalf("op=%d ReadBoolQ15: %v", opIndex, err)
		}
		if got != op.sym {
			t.Fatalf("op=%d bool=%d want %d", opIndex, got, op.sym)
		}
	case roundTripOpBit:
		got, err := r.ReadBit()
		if err != nil {
			t.Fatalf("op=%d ReadBit: %v", opIndex, err)
		}
		if got != op.sym {
			t.Fatalf("op=%d bit=%d want %d", opIndex, got, op.sym)
		}
	case roundTripOpLiteral:
		got, err := r.ReadBits(op.nbits)
		if err != nil {
			t.Fatalf("op=%d ReadBits(%d): %v", opIndex, op.nbits, err)
		}
		if got != op.value {
			t.Fatalf("op=%d literal=%#x want %#x", opIndex, got, op.value)
		}
	case roundTripOpFixedSymbol:
		row := fixed[op.row]
		cdf := append([]uint16(nil), row.values...)
		got, err := r.ReadSymbol(cdf, row.symbols)
		if err != nil {
			t.Fatalf("op=%d fixed ReadSymbol: %v", opIndex, err)
		}
		if got != int(op.sym) {
			t.Fatalf("op=%d fixed symbol=%d want %d", opIndex, got, op.sym)
		}
	case roundTripOpAdaptiveSymbol:
		roundTripReadSymbol(t, r, &cdfs.Rows[op.row], int(op.sym), opIndex, "adaptive")
	case roundTripOpCDF:
		roundTripReadSymbol(t, r, &cdfs.Rows[op.row], int(op.sym), opIndex, "cdf")
	case roundTripOpBinaryCDF:
		roundTripReadSymbol(t, r, &cdfs.Rows[roundTripRowBinary], int(op.sym), opIndex, "binary")
	case roundTripOpCDF3:
		roundTripReadSymbol(t, r, &cdfs.Rows[roundTripRowCDF3], int(op.sym), opIndex, "cdf3")
	case roundTripOpCDF4:
		roundTripReadSymbol(t, r, &cdfs.Rows[roundTripRowCDF4], int(op.sym), opIndex, "cdf4")
	case roundTripOpCDF4Zero:
		roundTripReadSymbol(t, r, &cdfs.Rows[roundTripRowCDF4], 0, opIndex, "cdf4-zero")
	case roundTripOpCDF5:
		roundTripReadSymbol(t, r, &cdfs.Rows[roundTripRowCDF5], int(op.sym), opIndex, "cdf5")
	case roundTripOpCDF7:
		roundTripReadSymbol(t, r, &cdfs.Rows[roundTripRowCDF7], int(op.sym), opIndex, "cdf7")
	case roundTripOpCoeffRun:
		roundTripDecodeCoeffRun(t, r, cdfs, op, opIndex)
	}
}

func roundTripRoundTrip(t *testing.T, tag string, rec roundTripRecord, ops []roundTripOp, initial roundTripCDFs, fixed []roundTripFixedCDF) {
	t.Helper()
	if rec.err != nil {
		t.Fatalf("%s: Finish: %v", tag, rec.err)
	}
	cdfs := initial
	r := NewReaderWithCDFUpdate(rec.bytes, true)
	for i, op := range ops {
		roundTripDecodeOp(t, &r, &cdfs, fixed, op, i)
	}
	if cdfs != rec.cdfs {
		t.Fatalf("%s decode CDF image mismatch: got %#v want %#v", tag, cdfs, rec.cdfs)
	}
}

func TestWriterReaderRoundTrip(t *testing.T) {
	streams := 24
	if testing.Short() {
		streams = 4
	}
	fixed := roundTripFixedCDFs(t)
	rng := rand.New(rand.NewSource(0xb0a51d))
	for shape := 0; shape < 4; shape++ {
		for stream := 0; stream < streams; stream++ {
			initial := roundTripSeedCDFs(t, rng, shape+stream)
			ops := roundTripBuildOps(rng, fixed, 96+rng.Intn(192))
			dstCap := rng.Intn(19)
			rec := roundTripEncodeStream(ops, initial, fixed, dstCap)
			tag := "shape=" + strconv.Itoa(shape) + "/stream=" + strconv.Itoa(stream) + "/dstCap=" + strconv.Itoa(dstCap)
			roundTripRoundTrip(t, tag, rec, ops, initial, fixed)
		}
	}
}
