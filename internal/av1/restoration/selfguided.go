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
	for row := -SGRProjBorderVert; row < height+SGRProjBorderVert; row++ {
		for col := -SGRProjBorderHorz; col < width+SGRProjBorderHorz; col++ {
			sample := src[srcOrigin+row*srcStride+col]
			if sample > max {
				return ErrInvalidRestoration
			}
			dgd[dgdOrigin+row*dgdStride+col] = int32(sample)
		}
	}

	params := SGRParameterTable[paramsIndex]
	if params.Radius[0] == 0 && params.Radius[1] == 0 {
		return ErrInvalidRestoration
	}
	if params.Radius[0] > 0 {
		selfguidedFast(dgd, dgdOrigin, width, height, dgdStride, flt0, width, int(bitDepth), paramsIndex, 0, aBuf, bBuf, bufStride)
	}
	if params.Radius[1] > 0 {
		selfguided(dgd, dgdOrigin, width, height, dgdStride, flt1, width, int(bitDepth), paramsIndex, 1, aBuf, bBuf, bufStride)
	}

	xq := DecodeSGRXQ(xqd, params)
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			k := row*width + col
			pre := int32(src[srcOrigin+row*srcStride+col])
			u := pre << SGRProjRstBits
			v := u << SGRProjPrjBits
			if params.Radius[0] > 0 {
				v += int32(xq[0]) * (flt0[k] - u)
			}
			if params.Radius[1] > 0 {
				v += int32(xq[1]) * (flt1[k] - u)
			}
			w := roundPowerOfTwo(v, SGRProjPrjBits+SGRProjRstBits)
			dst[row*dstStride+col] = uint16(clampInt32(w, 0, int32(max)))
		}
	}
	return nil
}

func selfguidedFast(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, dst []int32, dstStride int, bitDepth int, paramsIndex int, radiusIndex int, aBuf []int32, bBuf []int32, bufStride int) {
	calculateIntermediate(dgd, dgdOrigin, width, height, dgdStride, bitDepth, paramsIndex, radiusIndex, 1, aBuf, bBuf, bufStride)
	aOrigin := SGRProjBorderVert*bufStride + SGRProjBorderHorz
	for row := 0; row < height; row++ {
		if row&1 == 0 {
			for col := 0; col < width; col++ {
				k := aOrigin + row*bufStride + col
				l := dgdOrigin + row*dgdStride + col
				nb := 5
				a := (aBuf[k-bufStride]+aBuf[k+bufStride])*6 +
					(aBuf[k-1-bufStride]+aBuf[k-1+bufStride]+aBuf[k+1-bufStride]+aBuf[k+1+bufStride])*5
				b := (bBuf[k-bufStride]+bBuf[k+bufStride])*6 +
					(bBuf[k-1-bufStride]+bBuf[k-1+bufStride]+bBuf[k+1-bufStride]+bBuf[k+1+bufStride])*5
				dst[row*dstStride+col] = roundPowerOfTwo(a*dgd[l]+b, SGRProjSgrBits+nb-SGRProjRstBits)
			}
			continue
		}
		for col := 0; col < width; col++ {
			k := aOrigin + row*bufStride + col
			l := dgdOrigin + row*dgdStride + col
			nb := 4
			a := aBuf[k]*6 + (aBuf[k-1]+aBuf[k+1])*5
			b := bBuf[k]*6 + (bBuf[k-1]+bBuf[k+1])*5
			dst[row*dstStride+col] = roundPowerOfTwo(a*dgd[l]+b, SGRProjSgrBits+nb-SGRProjRstBits)
		}
	}
}

func selfguided(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, dst []int32, dstStride int, bitDepth int, paramsIndex int, radiusIndex int, aBuf []int32, bBuf []int32, bufStride int) {
	calculateIntermediate(dgd, dgdOrigin, width, height, dgdStride, bitDepth, paramsIndex, radiusIndex, 0, aBuf, bBuf, bufStride)
	aOrigin := SGRProjBorderVert*bufStride + SGRProjBorderHorz
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			k := aOrigin + row*bufStride + col
			l := dgdOrigin + row*dgdStride + col
			nb := 5
			a := (aBuf[k]+aBuf[k-1]+aBuf[k+1]+aBuf[k-bufStride]+aBuf[k+bufStride])*4 +
				(aBuf[k-1-bufStride]+aBuf[k-1+bufStride]+aBuf[k+1-bufStride]+aBuf[k+1+bufStride])*3
			b := (bBuf[k]+bBuf[k-1]+bBuf[k+1]+bBuf[k-bufStride]+bBuf[k+bufStride])*4 +
				(bBuf[k-1-bufStride]+bBuf[k-1+bufStride]+bBuf[k+1-bufStride]+bBuf[k+1+bufStride])*3
			dst[row*dstStride+col] = roundPowerOfTwo(a*dgd[l]+b, SGRProjSgrBits+nb-SGRProjRstBits)
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
	boxsum(dgd, srcOrigin, widthExt, heightExt, dgdStride, r, false, bBuf, bufStride)
	boxsum(dgd, srcOrigin, widthExt, heightExt, dgdStride, r, true, aBuf, bufStride)
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
			z := roundPowerOfTwoUnsigned(p*uint32(params.S[radiusIndex]), SGRProjMTableBits)
			if z > 255 {
				z = 255
			}
			aBuf[k] = xByXPlus1Value(z)
			bBuf[k] = int32(roundPowerOfTwoUnsigned(uint32(SGRProjSgr-int(aBuf[k]))*b*uint32(oneByX[n-1]), SGRProjRecipBits))
		}
	}
}

func boxsum(src []int32, srcOrigin int, width int, height int, srcStride int, radius int, squared bool, dst []int32, dstStride int) {
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			sum := int32(0)
			y0 := maxInt(0, row-radius)
			y1 := minInt(height-1, row+radius)
			x0 := maxInt(0, col-radius)
			x1 := minInt(width-1, col+radius)
			for y := y0; y <= y1; y++ {
				for x := x0; x <= x1; x++ {
					v := src[srcOrigin+y*srcStride+x]
					if squared {
						sum += v * v
					} else {
						sum += v
					}
				}
			}
			dst[row*dstStride+col] = sum
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
	return blockFitsAt(length, stride, origin-borderY*stride-borderX, width+2*borderX, height+2*borderY)
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
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
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
