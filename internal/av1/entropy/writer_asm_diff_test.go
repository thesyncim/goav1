// M-B0 writer-spine differential harness (see writer_asm_arm64_spec.go).
//
// There is no writer asm kernel in this slice. The harness still exposes both
// future dispatch positions and intentionally routes both through the pure-Go
// Writer, proving that the apparatus can compare complete writer records before
// B1/B2 add assembly. GOAV1_WRITER_DIFF scales the randomized stream count for
// soak runs.
package entropy

import (
	"bytes"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"testing"
)

type writerDiffVariant int

const (
	writerDiffKernelOn writerDiffVariant = iota
	writerDiffKernelOff
)

func (v writerDiffVariant) String() string {
	switch v {
	case writerDiffKernelOn:
		return "kernel-on"
	case writerDiffKernelOff:
		return "kernel-off"
	default:
		return "unknown"
	}
}

const (
	writerDiffRowBinary = iota
	writerDiffRowCDF3
	writerDiffRowCDF4
	writerDiffRowCDF5
	writerDiffRowCDF7
	writerDiffRowCDF8
	writerDiffRowCDF16
	writerDiffRows
)

var writerDiffRowSymbols = [writerDiffRows]int{2, 3, 4, 5, 7, 8, 16}

type writerDiffCDFs struct {
	Rows [writerDiffRows]CDF
}

type writerDiffFixedCDF struct {
	symbols int
	values  []uint16
}

type writerDiffWriterState struct {
	buf    []byte
	bufNil bool
	bufCap int
	offs   int
	low    uint64
	rng    uint32
	cnt    int32
	tell   int
	err    string
}

type writerDiffRecord struct {
	bytes        []byte
	tell         int
	beforeFinish writerDiffWriterState
	afterFinish  writerDiffWriterState
	cdfs         writerDiffCDFs
	err          string
}

type writerDiffOpKind uint8

const (
	writerDiffOpBoolQ15 writerDiffOpKind = iota
	writerDiffOpBit
	writerDiffOpLiteral
	writerDiffOpFixedSymbol
	writerDiffOpAdaptiveSymbol
	writerDiffOpCDF
	writerDiffOpBinaryCDF
	writerDiffOpCDF3
	writerDiffOpCDF4
	writerDiffOpCDF4Zero
	writerDiffOpCDF5
	writerDiffOpCDF7
	writerDiffOpCoeffRun
)

type writerDiffOp struct {
	kind  writerDiffOpKind
	row   uint8
	sym   uint8
	prob  uint16
	nbits uint8
	value uint32
	level uint16
	sign  uint8
	dc    bool
}

func writerDiffIters(defaultStreams int) int {
	if v := os.Getenv("GOAV1_WRITER_DIFF"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return defaultStreams
}

func writerDiffErrString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func writerDiffSnapshot(w *Writer) writerDiffWriterState {
	return writerDiffWriterState{
		buf:    append([]byte(nil), w.buf...),
		bufNil: w.buf == nil,
		bufCap: cap(w.buf),
		offs:   w.offs,
		low:    w.low,
		rng:    w.rng,
		cnt:    w.cnt,
		tell:   w.Tell(),
		err:    writerDiffErrString(w.err),
	}
}

func writerDiffStateEqual(a, b writerDiffWriterState) bool {
	return a.bufNil == b.bufNil &&
		a.bufCap == b.bufCap &&
		a.offs == b.offs &&
		a.low == b.low &&
		a.rng == b.rng &&
		a.cnt == b.cnt &&
		a.tell == b.tell &&
		a.err == b.err &&
		bytes.Equal(a.buf, b.buf)
}

func writerDiffCompareRecords(t *testing.T, tag string, aVariant, bVariant writerDiffVariant, a, b writerDiffRecord) {
	t.Helper()
	fail := func(field string, av, bv any) {
		t.Fatalf("%s %s vs %s: %s mismatch: %v vs %v", tag, aVariant, bVariant, field, av, bv)
	}
	if a.err != b.err {
		fail("finish error", a.err, b.err)
	}
	if a.tell != b.tell {
		fail("Tell", a.tell, b.tell)
	}
	if !writerDiffStateEqual(a.beforeFinish, b.beforeFinish) {
		fail("pre-Finish Writer state", a.beforeFinish, b.beforeFinish)
	}
	if !bytes.Equal(a.bytes, b.bytes) {
		fail("Finish bytes", a.bytes, b.bytes)
	}
	if !writerDiffStateEqual(a.afterFinish, b.afterFinish) {
		fail("post-Finish Writer state", a.afterFinish, b.afterFinish)
	}
	if a.cdfs != b.cdfs {
		fail("post-write CDF image", a.cdfs, b.cdfs)
	}
}

func writerDiffICDFFromWeights(t *testing.T, weights []int) []uint16 {
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

func writerDiffFixedCDFs(t *testing.T) []writerDiffFixedCDF {
	t.Helper()
	weights := [][]int{
		{1, 5},
		{3, 1, 4, 1},
		{2, 7, 1, 8, 2, 8, 1, 8},
		{1, 1, 2, 3, 5, 8, 13, 21, 3, 2, 1, 4, 6, 7, 9, 11},
	}
	out := make([]writerDiffFixedCDF, len(weights))
	for i, row := range weights {
		out[i] = writerDiffFixedCDF{
			symbols: len(row),
			values:  writerDiffICDFFromWeights(t, row),
		}
	}
	return out
}

func writerDiffSeedRow(t *testing.T, rng *rand.Rand, cdf *CDF, symbols, shape int) {
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

func writerDiffSeedCDFs(t *testing.T, rng *rand.Rand, shape int) writerDiffCDFs {
	t.Helper()
	var cdfs writerDiffCDFs
	for row, symbols := range writerDiffRowSymbols {
		writerDiffSeedRow(t, rng, &cdfs.Rows[row], symbols, shape+row)
	}
	return cdfs
}

func writerDiffRandomOp(rng *rand.Rand, fixed []writerDiffFixedCDF) writerDiffOp {
	switch rng.Intn(13) {
	case 0:
		return writerDiffOp{
			kind: writerDiffOpBoolQ15,
			sym:  uint8(rng.Intn(2)),
			prob: uint16(1 + rng.Intn(CDFProbTop-1)),
		}
	case 1:
		return writerDiffOp{kind: writerDiffOpBit, sym: uint8(rng.Intn(2))}
	case 2:
		n := uint8(1 + rng.Intn(24))
		return writerDiffOp{
			kind:  writerDiffOpLiteral,
			nbits: n,
			value: rng.Uint32() & ((uint32(1) << n) - 1),
		}
	case 3:
		row := uint8(rng.Intn(len(fixed)))
		return writerDiffOp{
			kind: writerDiffOpFixedSymbol,
			row:  row,
			sym:  uint8(rng.Intn(fixed[row].symbols)),
		}
	case 4:
		row := uint8(writerDiffRowCDF8 + rng.Intn(2))
		return writerDiffOp{
			kind: writerDiffOpAdaptiveSymbol,
			row:  row,
			sym:  uint8(rng.Intn(writerDiffRowSymbols[row])),
		}
	case 5:
		row := uint8(rng.Intn(writerDiffRows))
		return writerDiffOp{
			kind: writerDiffOpCDF,
			row:  row,
			sym:  uint8(rng.Intn(writerDiffRowSymbols[row])),
		}
	case 6:
		return writerDiffOp{kind: writerDiffOpBinaryCDF, sym: uint8(rng.Intn(2))}
	case 7:
		return writerDiffOp{kind: writerDiffOpCDF3, sym: uint8(rng.Intn(3))}
	case 8:
		if rng.Intn(4) == 0 {
			return writerDiffOp{kind: writerDiffOpCDF4Zero}
		}
		return writerDiffOp{kind: writerDiffOpCDF4, sym: uint8(rng.Intn(4))}
	case 9:
		return writerDiffOp{kind: writerDiffOpCDF5, sym: uint8(rng.Intn(5))}
	case 10:
		return writerDiffOp{kind: writerDiffOpCDF7, sym: uint8(rng.Intn(7))}
	default:
		level := 1 + rng.Intn(40)
		if rng.Intn(5) == 0 {
			level = 15 + rng.Intn(64)
		}
		return writerDiffOp{
			kind:  writerDiffOpCoeffRun,
			level: uint16(level),
			sign:  uint8(rng.Intn(2)),
			dc:    rng.Intn(3) == 0,
		}
	}
}

func writerDiffBuildOps(rng *rand.Rand, fixed []writerDiffFixedCDF, n int) []writerDiffOp {
	ops := make([]writerDiffOp, n)
	for i := range ops {
		ops[i] = writerDiffRandomOp(rng, fixed)
	}
	return ops
}

func writerDiffWriteGolomb(w *Writer, level int) {
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
	writerDiffNumBaseLevels  = 2
	writerDiffBaseRange      = 12
	writerDiffBRCDFSize      = 4
	writerDiffMaxBaseBRRange = 15
)

func writerDiffWriteBR(w *Writer, cdf *CDF, level int) {
	baseRange := level - 1 - writerDiffNumBaseLevels
	for idx := 0; idx < writerDiffBaseRange; idx += writerDiffBRCDFSize - 1 {
		k := min(baseRange-idx, writerDiffBRCDFSize-1)
		w.WriteCDF4(cdf, k)
		if k < writerDiffBRCDFSize-1 {
			break
		}
	}
}

func writerDiffWriteCoeffRun(w *Writer, cdfs *writerDiffCDFs, op writerDiffOp) {
	level := int(op.level)
	baseSym := min(level, 3) - 1
	w.WriteCDF3(&cdfs.Rows[writerDiffRowCDF3], baseSym)
	if level > writerDiffNumBaseLevels {
		writerDiffWriteBR(w, &cdfs.Rows[writerDiffRowCDF4], level)
	}
	if op.dc {
		w.WriteBinaryCDFTrusted(&cdfs.Rows[writerDiffRowBinary], int(op.sign))
	} else {
		w.WriteBit(int(op.sign))
	}
	if level >= writerDiffMaxBaseBRRange {
		writerDiffWriteGolomb(w, level-writerDiffMaxBaseBRRange)
	}
}

func writerDiffApplyOp(w *Writer, cdfs *writerDiffCDFs, fixed []writerDiffFixedCDF, op writerDiffOp) {
	switch op.kind {
	case writerDiffOpBoolQ15:
		w.WriteBoolQ15(int(op.sym), uint32(op.prob))
	case writerDiffOpBit:
		w.WriteBit(int(op.sym))
	case writerDiffOpLiteral:
		w.WriteLiteral(op.value, int(op.nbits))
	case writerDiffOpFixedSymbol:
		row := fixed[op.row]
		w.WriteSymbol(int(op.sym), row.values, row.symbols)
	case writerDiffOpAdaptiveSymbol:
		row := &cdfs.Rows[op.row]
		w.WriteSymbolAdaptive(int(op.sym), row.Values())
	case writerDiffOpCDF:
		w.WriteCDF(&cdfs.Rows[op.row], int(op.sym))
	case writerDiffOpBinaryCDF:
		w.WriteBinaryCDFTrusted(&cdfs.Rows[writerDiffRowBinary], int(op.sym))
	case writerDiffOpCDF3:
		w.WriteCDF3(&cdfs.Rows[writerDiffRowCDF3], int(op.sym))
	case writerDiffOpCDF4:
		w.WriteCDF4(&cdfs.Rows[writerDiffRowCDF4], int(op.sym))
	case writerDiffOpCDF4Zero:
		w.WriteCDF4Zero(&cdfs.Rows[writerDiffRowCDF4])
	case writerDiffOpCDF5:
		w.WriteCDF5(&cdfs.Rows[writerDiffRowCDF5], int(op.sym))
	case writerDiffOpCDF7:
		w.WriteCDF7(&cdfs.Rows[writerDiffRowCDF7], int(op.sym))
	case writerDiffOpCoeffRun:
		writerDiffWriteCoeffRun(w, cdfs, op)
	}
}

func writerDiffEncodeStream(variant writerDiffVariant, ops []writerDiffOp, initial writerDiffCDFs, fixed []writerDiffFixedCDF, dstCap int) writerDiffRecord {
	_ = variant // Both B0 switch positions intentionally run pure Go.
	cdfs := initial
	w := NewWriter(make([]byte, 0, dstCap))
	for _, op := range ops {
		writerDiffApplyOp(&w, &cdfs, fixed, op)
	}
	tell := w.Tell()
	before := writerDiffSnapshot(&w)
	buf, err := w.Finish()
	return writerDiffRecord{
		bytes:        append([]byte(nil), buf...),
		tell:         tell,
		beforeFinish: before,
		afterFinish:  writerDiffSnapshot(&w),
		cdfs:         cdfs,
		err:          writerDiffErrString(err),
	}
}

func writerDiffReadGolomb(t *testing.T, r *Reader, opIndex int) int {
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

func writerDiffReadSymbol(t *testing.T, r *Reader, cdf *CDF, want int, opIndex int, label string) {
	t.Helper()
	got, err := r.ReadSymbol(cdf.Values(), cdf.Symbols())
	if err != nil {
		t.Fatalf("op=%d %s ReadSymbol: %v", opIndex, label, err)
	}
	if got != want {
		t.Fatalf("op=%d %s symbol=%d want %d", opIndex, label, got, want)
	}
}

func writerDiffReadBR(t *testing.T, r *Reader, cdf *CDF, level, opIndex int) {
	t.Helper()
	baseRange := level - 1 - writerDiffNumBaseLevels
	for idx := 0; idx < writerDiffBaseRange; idx += writerDiffBRCDFSize - 1 {
		want := min(baseRange-idx, writerDiffBRCDFSize-1)
		writerDiffReadSymbol(t, r, cdf, want, opIndex, "coeff-br")
		if want < writerDiffBRCDFSize-1 {
			break
		}
	}
}

func writerDiffDecodeCoeffRun(t *testing.T, r *Reader, cdfs *writerDiffCDFs, op writerDiffOp, opIndex int) {
	t.Helper()
	level := int(op.level)
	writerDiffReadSymbol(t, r, &cdfs.Rows[writerDiffRowCDF3], min(level, 3)-1, opIndex, "coeff-base-eob")
	if level > writerDiffNumBaseLevels {
		writerDiffReadBR(t, r, &cdfs.Rows[writerDiffRowCDF4], level, opIndex)
	}
	if op.dc {
		writerDiffReadSymbol(t, r, &cdfs.Rows[writerDiffRowBinary], int(op.sign), opIndex, "coeff-dc-sign")
	} else {
		got, err := r.ReadBit()
		if err != nil {
			t.Fatalf("op=%d coeff-sign ReadBit: %v", opIndex, err)
		}
		if got != op.sign {
			t.Fatalf("op=%d coeff-sign=%d want %d", opIndex, got, op.sign)
		}
	}
	if level >= writerDiffMaxBaseBRRange {
		got := writerDiffReadGolomb(t, r, opIndex)
		want := level - writerDiffMaxBaseBRRange
		if got != want {
			t.Fatalf("op=%d coeff-golomb=%d want %d", opIndex, got, want)
		}
	}
}

func writerDiffDecodeOp(t *testing.T, r *Reader, cdfs *writerDiffCDFs, fixed []writerDiffFixedCDF, op writerDiffOp, opIndex int) {
	t.Helper()
	switch op.kind {
	case writerDiffOpBoolQ15:
		got, err := r.ReadBoolQ15(op.prob)
		if err != nil {
			t.Fatalf("op=%d ReadBoolQ15: %v", opIndex, err)
		}
		if got != op.sym {
			t.Fatalf("op=%d bool=%d want %d", opIndex, got, op.sym)
		}
	case writerDiffOpBit:
		got, err := r.ReadBit()
		if err != nil {
			t.Fatalf("op=%d ReadBit: %v", opIndex, err)
		}
		if got != op.sym {
			t.Fatalf("op=%d bit=%d want %d", opIndex, got, op.sym)
		}
	case writerDiffOpLiteral:
		got, err := r.ReadBits(op.nbits)
		if err != nil {
			t.Fatalf("op=%d ReadBits(%d): %v", opIndex, op.nbits, err)
		}
		if got != op.value {
			t.Fatalf("op=%d literal=%#x want %#x", opIndex, got, op.value)
		}
	case writerDiffOpFixedSymbol:
		row := fixed[op.row]
		cdf := append([]uint16(nil), row.values...)
		got, err := r.ReadSymbol(cdf, row.symbols)
		if err != nil {
			t.Fatalf("op=%d fixed ReadSymbol: %v", opIndex, err)
		}
		if got != int(op.sym) {
			t.Fatalf("op=%d fixed symbol=%d want %d", opIndex, got, op.sym)
		}
	case writerDiffOpAdaptiveSymbol:
		writerDiffReadSymbol(t, r, &cdfs.Rows[op.row], int(op.sym), opIndex, "adaptive")
	case writerDiffOpCDF:
		writerDiffReadSymbol(t, r, &cdfs.Rows[op.row], int(op.sym), opIndex, "cdf")
	case writerDiffOpBinaryCDF:
		writerDiffReadSymbol(t, r, &cdfs.Rows[writerDiffRowBinary], int(op.sym), opIndex, "binary")
	case writerDiffOpCDF3:
		writerDiffReadSymbol(t, r, &cdfs.Rows[writerDiffRowCDF3], int(op.sym), opIndex, "cdf3")
	case writerDiffOpCDF4:
		writerDiffReadSymbol(t, r, &cdfs.Rows[writerDiffRowCDF4], int(op.sym), opIndex, "cdf4")
	case writerDiffOpCDF4Zero:
		writerDiffReadSymbol(t, r, &cdfs.Rows[writerDiffRowCDF4], 0, opIndex, "cdf4-zero")
	case writerDiffOpCDF5:
		writerDiffReadSymbol(t, r, &cdfs.Rows[writerDiffRowCDF5], int(op.sym), opIndex, "cdf5")
	case writerDiffOpCDF7:
		writerDiffReadSymbol(t, r, &cdfs.Rows[writerDiffRowCDF7], int(op.sym), opIndex, "cdf7")
	case writerDiffOpCoeffRun:
		writerDiffDecodeCoeffRun(t, r, cdfs, op, opIndex)
	}
}

func writerDiffRoundTrip(t *testing.T, tag string, rec writerDiffRecord, ops []writerDiffOp, initial writerDiffCDFs, fixed []writerDiffFixedCDF) {
	t.Helper()
	if rec.err != "" {
		t.Fatalf("%s finish error before round-trip: %s", tag, rec.err)
	}
	cdfs := initial
	r := NewReaderWithCDFUpdate(rec.bytes, true)
	for i, op := range ops {
		writerDiffDecodeOp(t, &r, &cdfs, fixed, op, i)
	}
	if cdfs != rec.cdfs {
		t.Fatalf("%s decode CDF image mismatch: got %#v want %#v", tag, cdfs, rec.cdfs)
	}
}

func TestWriterSpineVariantsLockstep(t *testing.T) {
	streams := writerDiffIters(24)
	if testing.Short() {
		streams = 4
	}
	fixed := writerDiffFixedCDFs(t)
	rng := rand.New(rand.NewSource(0xb0a51d))
	for shape := 0; shape < 4; shape++ {
		for stream := 0; stream < streams; stream++ {
			initial := writerDiffSeedCDFs(t, rng, shape+stream)
			ops := writerDiffBuildOps(rng, fixed, 96+rng.Intn(192))
			dstCap := rng.Intn(19)
			on := writerDiffEncodeStream(writerDiffKernelOn, ops, initial, fixed, dstCap)
			off := writerDiffEncodeStream(writerDiffKernelOff, ops, initial, fixed, dstCap)
			tag := "shape=" + strconv.Itoa(shape) + "/stream=" + strconv.Itoa(stream) + "/dstCap=" + strconv.Itoa(dstCap)
			writerDiffCompareRecords(t, tag, writerDiffKernelOn, writerDiffKernelOff, on, off)
			writerDiffRoundTrip(t, tag+"/"+writerDiffKernelOn.String(), on, ops, initial, fixed)
			writerDiffRoundTrip(t, tag+"/"+writerDiffKernelOff.String(), off, ops, initial, fixed)
		}
	}
}
