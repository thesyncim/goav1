// Ported from libaom:
//   av1/common/restoration.c
//   av1/common/restoration.h
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package restoration

const (
	ProcUnitSize = 64

	SGRProjBorderVert = 3
	SGRProjBorderHorz = 3
	SGRProjParams     = 16

	SGRProjPrjBits = 7
	SGRProjRstBits = 4
	SGRProjSgrBits = 8
	SGRProjSgr     = 1 << SGRProjSgrBits

	SGRProjMTableBits = 20
	SGRProjRecipBits  = 12

	SGRProjPrjMin0    = -(1 << SGRProjPrjBits) * 3 / 4
	SGRProjPrjMax0    = SGRProjPrjMin0 + (1 << SGRProjPrjBits) - 1
	SGRProjPrjMin1    = -(1 << SGRProjPrjBits) / 4
	SGRProjPrjMax1    = SGRProjPrjMin1 + (1 << SGRProjPrjBits) - 1
	SGRProjPrjSubexpK = 4
)

type SGRParams struct {
	Radius [2]int
	S      [2]int
}

var SGRParameterTable = [SGRProjParams]SGRParams{
	{Radius: [2]int{2, 1}, S: [2]int{140, 3236}},
	{Radius: [2]int{2, 1}, S: [2]int{112, 2158}},
	{Radius: [2]int{2, 1}, S: [2]int{93, 1618}},
	{Radius: [2]int{2, 1}, S: [2]int{80, 1438}},
	{Radius: [2]int{2, 1}, S: [2]int{70, 1295}},
	{Radius: [2]int{2, 1}, S: [2]int{58, 1177}},
	{Radius: [2]int{2, 1}, S: [2]int{47, 1079}},
	{Radius: [2]int{2, 1}, S: [2]int{37, 996}},
	{Radius: [2]int{2, 1}, S: [2]int{30, 925}},
	{Radius: [2]int{2, 1}, S: [2]int{25, 863}},
	{Radius: [2]int{0, 1}, S: [2]int{-1, 2589}},
	{Radius: [2]int{0, 1}, S: [2]int{-1, 1618}},
	{Radius: [2]int{0, 1}, S: [2]int{-1, 1177}},
	{Radius: [2]int{0, 1}, S: [2]int{-1, 925}},
	{Radius: [2]int{2, 0}, S: [2]int{56, -1}},
	{Radius: [2]int{2, 0}, S: [2]int{22, -1}},
}

var xByXPlus1 = [256]int32{
	1, 128, 171, 192, 205, 213, 219, 224, 228, 230, 233, 235, 236, 238, 239,
	240, 241, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247, 247, 247,
	248, 248, 248, 248, 249, 249, 249, 249, 249, 250, 250, 250, 250, 250, 250,
	250, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 252, 252, 252, 252,
	252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 253, 253,
	253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253,
	253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 254, 254, 254,
	254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
	254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
	254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
	254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
	254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	256,
}

var oneByX = [25]int32{
	4096, 2048, 1365, 1024, 819, 683, 585, 512, 455, 410, 372, 341, 315,
	293, 273, 256, 241, 228, 216, 205, 195, 186, 178, 171, 164,
}

// boxsumImpl, selfguidedImpl and selfguidedFastImpl are the dispatch slots for
// the three SIMD-eligible SGR kernels. They are resolved once, at package init
// (see selfguided_dispatch_*.go), so the per-call cost is a single indirect
// call with no feature-detection branches. boxsum, selfguided and
// selfguidedFast below are the canonical bit-exact references; every tuned
// variant MUST match them sample for sample. The data-dependent LUT gather in
// calculateIntermediate (xByXPlus1 / oneByX) stays scalar in every build.
var (
	boxsumImpl         = boxsum
	selfguidedImpl     = selfguided
	selfguidedFastImpl = selfguidedFast
)

// SelfguidedScratchLen returns the int32 scratch length required by
// ApplySelfguidedRestoration for one processing unit.
func SelfguidedScratchLen(width int, height int) (int, error) {
	if !validSGRBlock(width, height) {
		return 0, ErrInvalidRestoration
	}
	dgdStride := width + 2*SGRProjBorderHorz
	dgdLen := dgdStride * (height + 2*SGRProjBorderVert)
	fltLen := width * height
	bufStride := sgrBufferStride(width)
	bufLen := bufStride * (height + 2*SGRProjBorderVert)
	return dgdLen + 2*fltLen + 2*bufLen, nil
}

// DecodeSGRXQ ports av1_decode_xq.
func DecodeSGRXQ(xqd [2]int, params SGRParams) [2]int {
	var xq [2]int
	switch {
	case params.Radius[0] == 0:
		xq[0] = 0
		xq[1] = (1 << SGRProjPrjBits) - xqd[1]
	case params.Radius[1] == 0:
		xq[0] = xqd[0]
		xq[1] = 0
	default:
		xq[0] = xqd[0]
		xq[1] = (1 << SGRProjPrjBits) - xq[0] - xqd[1]
	}
	return xq
}

// ApplySelfguidedRestoration ports av1_apply_selfguided_restoration_c for
// sample slices. srcOrigin identifies the top-left pixel of the unit and src
// must include a 3-pixel border around that origin.
func ApplySelfguidedRestoration(src []uint16, srcStride int, srcOrigin int, dst []uint16, dstStride int, width int, height int, paramsIndex int, xqd [2]int, bitDepth uint8, scratch []int32) error {
	max, err := maxSample(bitDepth)
	if err != nil {
		return err
	}
	if paramsIndex < 0 || paramsIndex >= len(SGRParameterTable) ||
		!validSGRBlock(width, height) ||
		!borderedBlockFits(len(src), srcStride, srcOrigin, width, height, SGRProjBorderHorz, SGRProjBorderVert) ||
		!blockFits(len(dst), dstStride, width, height) {
		return ErrInvalidRestoration
	}
	need, err := SelfguidedScratchLen(width, height)
	if err != nil || len(scratch) < need {
		return ErrInvalidRestoration
	}

	dgdStride := width + 2*SGRProjBorderHorz
	dgdLen := dgdStride * (height + 2*SGRProjBorderVert)
	fltLen := width * height
	bufStride := sgrBufferStride(width)
	bufLen := bufStride * (height + 2*SGRProjBorderVert)

	dgd := scratch[:dgdLen]
	flt0 := scratch[dgdLen : dgdLen+fltLen]
	flt1 := scratch[dgdLen+fltLen : dgdLen+2*fltLen]
	ab := scratch[dgdLen+2*fltLen:]
	aBuf := ab[:bufLen]
	bBuf := ab[bufLen : 2*bufLen]
	clearInt32s(dgd)
	clearInt32s(flt0)
	clearInt32s(flt1)
	clearInt32s(aBuf)
	clearInt32s(bBuf)

	dgdOrigin := SGRProjBorderVert*dgdStride + SGRProjBorderHorz
	extWidth := width + 2*SGRProjBorderHorz
	for row := -SGRProjBorderVert; row < height+SGRProjBorderVert; row++ {
		srcStart := srcOrigin + row*srcStride - SGRProjBorderHorz
		dgdStart := dgdOrigin + row*dgdStride - SGRProjBorderHorz
		srcRow := src[srcStart : srcStart+extWidth]
		dgdRow := dgd[dgdStart : dgdStart+extWidth]
		for col, sample := range srcRow {
			if sample > max {
				return ErrInvalidRestoration
			}
			dgdRow[col] = int32(sample)
		}
	}

	params := SGRParameterTable[paramsIndex]
	if params.Radius[0] == 0 && params.Radius[1] == 0 {
		return ErrInvalidRestoration
	}
	if params.Radius[0] > 0 {
		selfguidedFastImpl(dgd, dgdOrigin, width, height, dgdStride, flt0, width, int(bitDepth), paramsIndex, 0, aBuf, bBuf, bufStride)
	}
	if params.Radius[1] > 0 {
		selfguidedImpl(dgd, dgdOrigin, width, height, dgdStride, flt1, width, int(bitDepth), paramsIndex, 1, aBuf, bBuf, bufStride)
	}

	xq := DecodeSGRXQ(xqd, params)
	xq0 := int32(xq[0])
	xq1 := int32(xq[1])
	use0 := params.Radius[0] > 0
	use1 := params.Radius[1] > 0
	maxI := int32(max)
	for row := range height {
		srcRow := src[srcOrigin+row*srcStride : srcOrigin+row*srcStride+width]
		dstRow := dst[row*dstStride : row*dstStride+width]
		f0Row := flt0[row*width : row*width+width]
		f1Row := flt1[row*width : row*width+width]
		for col, s := range srcRow {
			u := int32(s) << SGRProjRstBits
			v := u << SGRProjPrjBits
			if use0 {
				v += xq0 * (f0Row[col] - u)
			}
			if use1 {
				v += xq1 * (f1Row[col] - u)
			}
			// libaom casts the rounded projection to int16_t before clipping,
			// so a value beyond [-32768, 32767] wraps modulo 2^16 and is then
			// re-extended to int32 inside clip_pixel_highbd. Match that exactly
			// instead of clamping the wider int32 directly, otherwise extreme
			// projections (very negative or very large) diverge by entire-pixel
			// values.
			rounded := roundPowerOfTwo(v, SGRProjPrjBits+SGRProjRstBits)
			w := int32(int16(rounded))
			dstRow[col] = uint16(clampInt32(w, 0, maxI))
		}
	}
	return nil
}

func selfguidedFast(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, dst []int32, dstStride int, bitDepth int, paramsIndex int, radiusIndex int, aBuf []int32, bBuf []int32, bufStride int) {
	calculateIntermediate(dgd, dgdOrigin, width, height, dgdStride, bitDepth, paramsIndex, radiusIndex, 1, aBuf, bBuf, bufStride)
	aOrigin := SGRProjBorderVert*bufStride + SGRProjBorderHorz
	const shiftEven = SGRProjSgrBits + 5 - SGRProjRstBits
	const shiftOdd = SGRProjSgrBits + 4 - SGRProjRstBits
	for row := range height {
		k0 := aOrigin + row*bufStride
		dgdRow := dgd[dgdOrigin+row*dgdStride : dgdOrigin+row*dgdStride+width]
		dstRow := dst[row*dstStride : row*dstStride+width]
		if row&1 == 0 {
			aPrev := aBuf[k0-bufStride-1 : k0-bufStride+width+1]
			aNext := aBuf[k0+bufStride-1 : k0+bufStride+width+1]
			bPrev := bBuf[k0-bufStride-1 : k0-bufStride+width+1]
			bNext := bBuf[k0+bufStride-1 : k0+bufStride+width+1]
			for col := range width {
				j := col + 1
				a := (aPrev[j]+aNext[j])*6 +
					(aPrev[j-1]+aPrev[j+1]+aNext[j-1]+aNext[j+1])*5
				b := (bPrev[j]+bNext[j])*6 +
					(bPrev[j-1]+bPrev[j+1]+bNext[j-1]+bNext[j+1])*5
				dstRow[col] = roundPowerOfTwo(a*dgdRow[col]+b, shiftEven)
			}
			continue
		}
		aCur := aBuf[k0-1 : k0+width+1]
		bCur := bBuf[k0-1 : k0+width+1]
		for col := range width {
			j := col + 1
			a := aCur[j]*6 + (aCur[j-1]+aCur[j+1])*5
			b := bCur[j]*6 + (bCur[j-1]+bCur[j+1])*5
			dstRow[col] = roundPowerOfTwo(a*dgdRow[col]+b, shiftOdd)
		}
	}
}

func selfguided(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, dst []int32, dstStride int, bitDepth int, paramsIndex int, radiusIndex int, aBuf []int32, bBuf []int32, bufStride int) {
	calculateIntermediate(dgd, dgdOrigin, width, height, dgdStride, bitDepth, paramsIndex, radiusIndex, 0, aBuf, bBuf, bufStride)
	aOrigin := SGRProjBorderVert*bufStride + SGRProjBorderHorz
	const nb = 5
	const shift = SGRProjSgrBits + nb - SGRProjRstBits
	for row := range height {
		k0 := aOrigin + row*bufStride
		// Reslice the three stencil rows so that index j maps to column j and
		// j-1/j+1 stay in bounds; the windows start one column left of k0.
		aPrev := aBuf[k0-bufStride-1 : k0-bufStride+width+1]
		aCur := aBuf[k0-1 : k0+width+1]
		aNext := aBuf[k0+bufStride-1 : k0+bufStride+width+1]
		bPrev := bBuf[k0-bufStride-1 : k0-bufStride+width+1]
		bCur := bBuf[k0-1 : k0+width+1]
		bNext := bBuf[k0+bufStride-1 : k0+bufStride+width+1]
		dgdRow := dgd[dgdOrigin+row*dgdStride : dgdOrigin+row*dgdStride+width]
		dstRow := dst[row*dstStride : row*dstStride+width]
		for col := range width {
			j := col + 1 // center index inside the +1-padded windows
			a := (aCur[j]+aCur[j-1]+aCur[j+1]+aPrev[j]+aNext[j])*4 +
				(aPrev[j-1]+aNext[j-1]+aPrev[j+1]+aNext[j+1])*3
			b := (bCur[j]+bCur[j-1]+bCur[j+1]+bPrev[j]+bNext[j])*4 +
				(bPrev[j-1]+bNext[j-1]+bPrev[j+1]+bNext[j+1])*3
			dstRow[col] = roundPowerOfTwo(a*dgdRow[col]+b, shift)
		}
	}
}

func calculateIntermediate(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, bitDepth int, paramsIndex int, radiusIndex int, pass int, aBuf []int32, bBuf []int32, bufStride int) {
	clearInt32s(aBuf)
	clearInt32s(bBuf)
	params := SGRParameterTable[paramsIndex]
	r := params.Radius[radiusIndex]
	widthExt := width + 2*SGRProjBorderHorz
	heightExt := height + 2*SGRProjBorderVert
	srcOrigin := dgdOrigin - SGRProjBorderVert*dgdStride - SGRProjBorderHorz
	boxsumImpl(dgd, srcOrigin, widthExt, heightExt, dgdStride, r, false, bBuf, bufStride)
	boxsumImpl(dgd, srcOrigin, widthExt, heightExt, dgdStride, r, true, aBuf, bufStride)
	aOrigin := SGRProjBorderVert*bufStride + SGRProjBorderHorz
	step := 1
	if pass != 0 {
		step = 2
	}
	n := (2*r + 1) * (2*r + 1)
	for row := -1; row < height+1; row += step {
		for col := -1; col < width+1; col++ {
			k := aOrigin + row*bufStride + col
			a := uint32(roundPowerOfTwo(aBuf[k], 2*(bitDepth-8)))
			b := uint32(roundPowerOfTwo(bBuf[k], bitDepth-8))
			p := uint32(0)
			if a*uint32(n) >= b*b {
				p = a*uint32(n) - b*b
			}
			z := min(roundPowerOfTwoUnsigned(p*uint32(params.S[radiusIndex]), SGRProjMTableBits), 255)
			aBuf[k] = xByXPlus1Value(z)
			// libaom uses the raw (unshifted) B[k] in the post-update, NOT the
			// bit-depth-scaled local `b`. Using the scaled b mistakenly scales
			// the SGR smoothed image by an extra 1<<(bit_depth-8) factor for
			// 10/12-bit input, producing the q63 first-frame divergence in flt
			// even when reconstruction and CDEF are byte-exact.
			bBuf[k] = int32(roundPowerOfTwoUnsigned(uint32(SGRProjSgr-int(aBuf[k]))*uint32(bBuf[k])*uint32(oneByX[n-1]), SGRProjRecipBits))
		}
	}
}

func boxsum(src []int32, srcOrigin int, width int, height int, srcStride int, radius int, squared bool, dst []int32, dstStride int) {
	for row := range height {
		y0 := maxInt(0, row-radius)
		y1 := minInt(height-1, row+radius)
		dstRow := dst[row*dstStride : row*dstStride+width]
		for col := range width {
			sum := int32(0)
			x0 := maxInt(0, col-radius)
			x1 := minInt(width-1, col+radius)
			if squared {
				for y := y0; y <= y1; y++ {
					base := srcOrigin + y*srcStride
					srcRow := src[base+x0 : base+x1+1]
					for _, v := range srcRow {
						sum += v * v
					}
				}
			} else {
				for y := y0; y <= y1; y++ {
					base := srcOrigin + y*srcStride
					srcRow := src[base+x0 : base+x1+1]
					for _, v := range srcRow {
						sum += v
					}
				}
			}
			dstRow[col] = sum
		}
	}
}

func xByXPlus1Value(z uint32) int32 {
	if z > 255 {
		z = 255
	}
	return xByXPlus1[z]
}

func minInt(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a int, b int) int {
	if a > b {
		return a
	}
	return b
}

func validSGRBlock(width int, height int) bool {
	return width > 0 && height > 0 && width <= ProcUnitSize && height <= ProcUnitSize
}

func sgrBufferStride(width int) int {
	widthExt := width + 2*SGRProjBorderHorz
	return ((widthExt + 3) &^ 3) + 16
}

func maxSample(bitDepth uint8) (uint16, error) {
	switch bitDepth {
	case 8, 10, 12:
		return uint16((1 << bitDepth) - 1), nil
	default:
		return 0, ErrInvalidRestoration
	}
}

func borderedBlockFits(length int, stride int, origin int, width int, height int, borderX int, borderY int) bool {
	if stride <= 0 || borderX < 0 || borderY < 0 || width <= 0 || height <= 0 {
		return false
	}
	// Compute origin-borderY*stride-borderX with overflow-checked arithmetic
	// so a hostile origin/stride combination cannot wrap the comparison
	// inside blockFitsAt.
	borderRows, ok := checkedMul(borderY, stride)
	if !ok {
		return false
	}
	originLessRows, ok := checkedSub(origin, borderRows)
	if !ok {
		return false
	}
	startOrigin, ok := checkedSub(originLessRows, borderX)
	if !ok {
		return false
	}
	totalWidth, ok := checkedAdd(width, 2*borderX)
	if !ok {
		return false
	}
	totalHeight, ok := checkedAdd(height, 2*borderY)
	if !ok {
		return false
	}
	return blockFitsAt(length, stride, startOrigin, totalWidth, totalHeight)
}

func checkedSub(a int, b int) (int, bool) {
	c := a - b
	if b > 0 && c > a {
		return 0, false
	}
	if b < 0 && c < a {
		return 0, false
	}
	return c, true
}

func blockFits(length int, stride int, width int, height int) bool {
	return blockFitsAt(length, stride, 0, width, height)
}

func blockFitsAt(length int, stride int, origin int, width int, height int) bool {
	if origin < 0 || stride <= 0 || width <= 0 || height <= 0 || stride < width {
		return false
	}
	lastRow, ok := checkedMul(height-1, stride)
	if !ok {
		return false
	}
	last, ok := checkedAdd(origin, lastRow)
	if !ok {
		return false
	}
	needed, ok := checkedAdd(last, width)
	return ok && needed <= length
}

func roundPowerOfTwo(v int32, bits int) int32 {
	if bits <= 0 {
		return v
	}
	return (v + int32(1<<(bits-1))) >> bits
}

func roundPowerOfTwoUnsigned(v uint32, bits int) uint32 {
	if bits <= 0 {
		return v
	}
	return (v + uint32(1<<(bits-1))) >> bits
}

func clampInt32(v int32, lo int32, hi int32) int32 {
	return min(max(v, lo), hi)
}

func clearInt32s(values []int32) {
	for i := range values {
		values[i] = 0
	}
}

func checkedAdd(a int, b int) (int, bool) {
	c := a + b
	if c < a {
		return 0, false
	}
	return c, true
}

func checkedMul(a int, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > int(^uint(0)>>1)/a {
		return 0, false
	}
	return a * b, true
}
