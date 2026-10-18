package tile

import (
	"fmt"
	"reflect"
	"testing"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

// The oracle bodies below freeze MarkInter and MarkInterMotion from clean
// main 45dc043b (unchanged since 8c68bb81). The MarkInterMotion oracle calls the frozen MarkInter
// body directly so it remains independent of the candidate helper.
func frozenMarkInterOracle(c *BlockModeContext, size BlockSize, x4, y4 int, result InterReferencesResult, hasChroma bool) error {
	if c == nil {
		return ErrInvalidDecodeState
	}
	dims, ok := size.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	if x4 < 0 || y4 < 0 ||
		x4+int(dims.W4) > MaxBlockModeSlots ||
		y4+int(dims.H4) > MaxBlockModeSlots {
		return ErrInvalidDecodeState
	}
	if !result.Ref[0].Valid() {
		return ErrInvalidDecodeState
	}
	if result.Compound {
		if !result.Ref[1].Valid() {
			return ErrInvalidDecodeState
		}
	} else if result.Ref[1] != ReferenceFrameNone {
		return ErrInvalidDecodeState
	}

	compound := boolByte(result.Compound)
	compIndex := uint8(1)
	w4 := int(dims.W4)
	h4 := int(dims.H4)
	xEnd := x4 + w4
	yEnd := y4 + h4
	clear(c.AboveIntra[x4:xEnd])
	clear(c.AboveCompGroup[x4:xEnd])
	clear(c.AboveInterIntra[x4:xEnd])
	clear(c.AboveMotionValid[x4:xEnd])
	clear(c.AboveInterpValid[x4:xEnd])
	clear(c.LeftIntra[y4:yEnd])
	clear(c.LeftCompGroup[y4:yEnd])
	clear(c.LeftInterIntra[y4:yEnd])
	clear(c.LeftMotionValid[y4:yEnd])
	clear(c.LeftInterpValid[y4:yEnd])
	if hasChroma {
		clear(c.AboveChromaIntra[x4:xEnd])
		clear(c.AboveChromaMode[x4:xEnd])
		clear(c.LeftChromaIntra[y4:yEnd])
		clear(c.LeftChromaMode[y4:yEnd])
	}
	for i := 0; i < w4; i++ {
		c.AboveRef[0][x4+i] = result.Ref[0]
		c.AboveRef[1][x4+i] = result.Ref[1]
		c.AboveCompound[x4+i] = compound
		c.AboveCompIndex[x4+i] = compIndex
		c.AboveBlockSize[x4+i] = size
	}
	for i := 0; i < h4; i++ {
		c.LeftRef[0][y4+i] = result.Ref[0]
		c.LeftRef[1][y4+i] = result.Ref[1]
		c.LeftCompound[y4+i] = compound
		c.LeftCompIndex[y4+i] = compIndex
		c.LeftBlockSize[y4+i] = size
	}
	c.clearGridInterMotion(size, x4, y4, dims)
	return nil
}

func frozenMarkInterMotionOracle(c *BlockModeContext, size BlockSize, x4, y4 int, result InterMotionResult, hasChroma bool) error {
	if err := validateInterMotion(result); err != nil {
		return err
	}
	if err := frozenMarkInterOracle(c, size, x4, y4, result.References, hasChroma); err != nil {
		return err
	}
	dims, ok := size.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	for i := 0; i < int(dims.W4); i++ {
		c.AboveInterMotion[x4+i] = result
		c.AboveMotionValid[x4+i] = 1
		c.AboveBlockSize[x4+i] = size
	}
	for i := 0; i < int(dims.H4); i++ {
		c.LeftInterMotion[y4+i] = result
		c.LeftMotionValid[y4+i] = 1
		c.LeftBlockSize[y4+i] = size
	}
	c.markGridInterMotion(size, x4, y4, result, dims)
	return nil
}

func seededBlockModeContext(t *testing.T, seed byte) BlockModeContext {
	t.Helper()
	typ := reflect.TypeOf(BlockModeContext{})
	if blockModeContextHasPointers(typ) {
		t.Fatal("BlockModeContext must remain pointer-free for the poisoned-context oracle")
	}
	var ctx BlockModeContext
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(&ctx)), int(unsafe.Sizeof(ctx)))
	for i := range bytes {
		bytes[i] = byte((i*37+int(seed))%255 + 1)
	}
	boolIndex := 0
	setContextBools(typ, unsafe.Pointer(&ctx), &boolIndex)
	return ctx
}

func blockModeContextHasPointers(typ reflect.Type) bool {
	switch typ.Kind() {
	case reflect.Pointer, reflect.Slice, reflect.Map, reflect.Chan, reflect.Func, reflect.Interface, reflect.String, reflect.UnsafePointer:
		return true
	case reflect.Array:
		return blockModeContextHasPointers(typ.Elem())
	case reflect.Struct:
		for i := 0; i < typ.NumField(); i++ {
			if blockModeContextHasPointers(typ.Field(i).Type) {
				return true
			}
		}
	}
	return false
}

func setContextBools(typ reflect.Type, ptr unsafe.Pointer, index *int) {
	switch typ.Kind() {
	case reflect.Bool:
		reflect.NewAt(typ, ptr).Elem().SetBool(*index&1 == 0)
		*index = *index + 1
	case reflect.Array:
		for i := 0; i < typ.Len(); i++ {
			setContextBools(typ.Elem(), unsafe.Add(ptr, uintptr(i)*typ.Elem().Size()), index)
		}
	case reflect.Struct:
		for i := 0; i < typ.NumField(); i++ {
			field := typ.Field(i)
			setContextBools(field.Type, unsafe.Add(ptr, field.Offset), index)
		}
	}
}

func validInterMotionForOracle(refs InterReferencesResult, n int) InterMotionResult {
	result := InterMotionResult{
		MV: [2]motion.Vector{
			{Row: int16(n*3 + 1), Col: int16(-n*5 - 2)},
			{Row: int16(n*7 + 3), Col: int16(-n*11 - 4)},
		},
		References: refs,
		InterIntra: n&1 != 0,
	}
	if refs.Compound {
		result.Mode = InterModeResult{Compound: true, CompoundMode: CompoundInterModeNewNew}
	} else {
		result.Mode = InterModeResult{Mode: InterModeNewMV}
	}
	return result
}

func positionsForInterContextOracle(size BlockSize) [][2]int {
	dims, _ := size.Dimensions()
	maxX := MaxBlockModeSlots - int(dims.W4)
	maxY := MaxBlockModeSlots - int(dims.H4)
	return [][2]int{{0, 0}, {maxX, maxY}, {maxX / 2, maxY / 2}}
}

func compareInterContextToFrozenOracle(t *testing.T, size BlockSize, x4, y4 int, refs InterReferencesResult, hasChroma bool, caseIndex int) {
	t.Helper()
	initial := seededBlockModeContext(t, byte(caseIndex*13+7))

	gotRefs, wantRefs := initial, initial
	gotRefsErr := gotRefs.MarkInter(size, x4, y4, refs, hasChroma)
	wantRefsErr := frozenMarkInterOracle(&wantRefs, size, x4, y4, refs, hasChroma)
	if gotRefsErr != wantRefsErr {
		t.Fatalf("MarkInter size=%d at=(%d,%d) refs=%+v chroma=%t err=%v want %v", size, x4, y4, refs, hasChroma, gotRefsErr, wantRefsErr)
	}
	if !reflect.DeepEqual(gotRefs, wantRefs) {
		t.Fatalf("MarkInter state differs from frozen oracle: size=%d at=(%d,%d) refs=%+v chroma=%t", size, x4, y4, refs, hasChroma)
	}

	motionResult := validInterMotionForOracle(refs, caseIndex)
	gotMotion, wantMotion := initial, initial
	gotMotionErr := gotMotion.MarkInterMotion(size, x4, y4, motionResult, hasChroma)
	wantMotionErr := frozenMarkInterMotionOracle(&wantMotion, size, x4, y4, motionResult, hasChroma)
	if gotMotionErr != wantMotionErr {
		t.Fatalf("MarkInterMotion size=%d at=(%d,%d) motion=%+v chroma=%t err=%v want %v", size, x4, y4, motionResult, hasChroma, gotMotionErr, wantMotionErr)
	}
	if !reflect.DeepEqual(gotMotion, wantMotion) {
		t.Fatalf("MarkInterMotion state differs from frozen oracle: size=%d at=(%d,%d) motion=%+v chroma=%t", size, x4, y4, motionResult, hasChroma)
	}
}

func TestBlockModeContextMarkInterSingleFillMatchesFrozenOracle(t *testing.T) {
	caseIndex := 0
	for size := BlockSize(0); size < blockSizeCount; size++ {
		positions := positionsForInterContextOracle(size)
		for positionIndex, pos := range positions {
			pairIndex := int(size)*len(positions) + positionIndex
			for _, hasChroma := range [...]bool{false, true} {
				single := InterReferencesResult{
					Ref:    [2]ReferenceFrame{ReferenceFrame(pairIndex % int(referenceFrameCount)), ReferenceFrameNone},
					Unidir: hasChroma,
				}
				compareInterContextToFrozenOracle(t, size, pos[0], pos[1], single, hasChroma, caseIndex)
				caseIndex++

				compound := InterReferencesResult{
					Ref:      [2]ReferenceFrame{ReferenceFrame((pairIndex / int(referenceFrameCount)) % int(referenceFrameCount)), ReferenceFrame(pairIndex % int(referenceFrameCount))},
					Compound: true,
					Unidir:   hasChroma,
				}
				compareInterContextToFrozenOracle(t, size, pos[0], pos[1], compound, hasChroma, caseIndex)
				caseIndex++
			}
		}
	}
}

func TestBlockModeContextMarkInterSingleFillPreservesValidationSequence(t *testing.T) {
	validRefs := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	invalidRefs := []struct {
		name string
		size BlockSize
		x4   int
		y4   int
		refs InterReferencesResult
	}{
		{name: "invalid size", size: blockSizeCount, refs: validRefs},
		{name: "negative x", size: BlockSize4x4, x4: -1, refs: validRefs},
		{name: "negative y", size: BlockSize4x4, y4: -1, refs: validRefs},
		{name: "x past edge", size: BlockSize4x4, x4: MaxBlockModeSlots, refs: validRefs},
		{name: "y past edge", size: BlockSize4x4, y4: MaxBlockModeSlots, refs: validRefs},
		{name: "wide block past edge", size: BlockSize64x64, x4: 17, refs: validRefs},
		{name: "tall block past edge", size: BlockSize64x64, y4: 17, refs: validRefs},
		{name: "missing first ref", size: BlockSize4x4, refs: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameNone, ReferenceFrameNone}}},
		{name: "compound missing second ref", size: BlockSize4x4, refs: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}, Compound: true}},
		{name: "single has second ref", size: BlockSize4x4, refs: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameGolden}}},
		{name: "edge and ref invalid", size: BlockSize4x4, x4: MaxBlockModeSlots, refs: InterReferencesResult{}},
	}
	for i, tc := range invalidRefs {
		t.Run(tc.name, func(t *testing.T) {
			initial := seededBlockModeContext(t, byte(i+41))
			got, want := initial, initial
			gotErr := got.MarkInter(tc.size, tc.x4, tc.y4, tc.refs, true)
			wantErr := frozenMarkInterOracle(&want, tc.size, tc.x4, tc.y4, tc.refs, true)
			if gotErr != wantErr || gotErr != ErrInvalidDecodeState {
				t.Fatalf("MarkInter error=%v oracle=%v want %v", gotErr, wantErr, ErrInvalidDecodeState)
			}
			if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(got, initial) {
				t.Fatal("invalid MarkInter changed context or diverged from frozen oracle")
			}
		})
	}

	validMotion := validInterMotionForOracle(validRefs, 5)
	invalidMotions := []struct {
		name   string
		size   BlockSize
		x4     int
		y4     int
		result InterMotionResult
	}{
		{name: "invalid size", size: blockSizeCount, result: validMotion},
		{name: "negative x", size: BlockSize4x4, x4: -1, result: validMotion},
		{name: "negative y", size: BlockSize4x4, y4: -1, result: validMotion},
		{name: "x past edge", size: BlockSize4x4, x4: MaxBlockModeSlots, result: validMotion},
		{name: "mode compound mismatch", size: BlockSize4x4, result: InterMotionResult{References: validRefs, Mode: InterModeResult{Compound: true, CompoundMode: CompoundInterModeNewNew}}},
		{name: "invalid single mode", size: BlockSize4x4, result: InterMotionResult{References: validRefs, Mode: InterModeResult{Mode: interModeCount}}},
		{name: "invalid compound mode", size: BlockSize4x4, result: InterMotionResult{References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameGolden}, Compound: true}, Mode: InterModeResult{Compound: true, CompoundMode: compoundInterModeCount}}},
		{name: "invalid motion ref with edge", size: BlockSize4x4, x4: MaxBlockModeSlots, result: InterMotionResult{References: InterReferencesResult{}, Mode: InterModeResult{Mode: InterModeNewMV}}},
	}
	for i, tc := range invalidMotions {
		t.Run("motion/"+tc.name, func(t *testing.T) {
			initial := seededBlockModeContext(t, byte(i+91))
			got, want := initial, initial
			gotErr := got.MarkInterMotion(tc.size, tc.x4, tc.y4, tc.result, false)
			wantErr := frozenMarkInterMotionOracle(&want, tc.size, tc.x4, tc.y4, tc.result, false)
			if gotErr != wantErr || gotErr != ErrInvalidDecodeState {
				t.Fatalf("MarkInterMotion error=%v oracle=%v want %v", gotErr, wantErr, ErrInvalidDecodeState)
			}
			if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(got, initial) {
				t.Fatal("invalid MarkInterMotion changed context or diverged from frozen oracle")
			}
		})
	}

	if got := (*BlockModeContext)(nil).MarkInter(BlockSize4x4, 0, 0, validRefs, false); got != frozenMarkInterOracle(nil, BlockSize4x4, 0, 0, validRefs, false) || got != ErrInvalidDecodeState {
		t.Fatalf("nil MarkInter error=%v", got)
	}
	if got := (*BlockModeContext)(nil).MarkInterMotion(BlockSize4x4, 0, 0, validMotion, false); got != frozenMarkInterMotionOracle(nil, BlockSize4x4, 0, 0, validMotion, false) || got != ErrInvalidDecodeState {
		t.Fatalf("nil MarkInterMotion error=%v", got)
	}
	invalidMotion := validMotion
	invalidMotion.Mode.Mode = interModeCount
	if got := (*BlockModeContext)(nil).MarkInterMotion(BlockSize4x4, 0, 0, invalidMotion, false); got != frozenMarkInterMotionOracle(nil, BlockSize4x4, 0, 0, invalidMotion, false) || got != ErrInvalidDecodeState {
		t.Fatalf("invalid motion before nil receiver error=%v", got)
	}
}

// Freeze the original filter commit as well: the combined candidate must not
// serve as its own motion/filter state oracle. The unchanged grid helpers only
// support the frozen original motion setup.
func frozenMarkInterFiltersOracle(c *BlockModeContext, size BlockSize, x4 int, y4 int, refs InterReferencesResult, filters motion.InterpFilters) error {
	if c == nil || !refs.Ref[0].Valid() || !filters.X.Valid() || !filters.Y.Valid() {
		return ErrInvalidDecodeState
	}
	if refs.Compound {
		if !refs.Ref[1].Valid() {
			return ErrInvalidDecodeState
		}
	} else if refs.Ref[1] != ReferenceFrameNone {
		return ErrInvalidDecodeState
	}
	dims, ok := size.Dimensions()
	if !ok || x4 < 0 || y4 < 0 ||
		x4+int(dims.W4) > MaxBlockModeSlots ||
		y4+int(dims.H4) > MaxBlockModeSlots {
		return ErrInvalidDecodeState
	}
	for i := 0; i < int(dims.W4); i++ {
		c.AboveInterp[x4+i] = filters
		c.AboveInterpValid[x4+i] = 1
	}
	for i := 0; i < int(dims.H4); i++ {
		c.LeftInterp[y4+i] = filters
		c.LeftInterpValid[y4+i] = 1
	}
	// Record the per-MI filter pair so the sub8x8 chroma predictor can read
	// each covered luma sub-block's own filters (libaom's this_mbmi->
	// interp_filters), mirroring the GridInterMotion grid written by
	// MarkInterMotion.
	xEnd := x4 + int(dims.W4)
	i0 := c.GridInterp[y4][x4:xEnd]
	v0 := c.GridInterpValid[y4][x4:xEnd]
	fillGridRow(i0, filters)
	fillGridRow(v0, 1)
	for y := y4 + 1; y < y4+int(dims.H4); y++ {
		copy(c.GridInterp[y][x4:xEnd], i0)
		copy(c.GridInterpValid[y][x4:xEnd], v0)
	}
	return nil
}

func frozenDecodedInterContextOracle(c *BlockModeContext, size BlockSize, x4, y4 int, result InterMotionResult, filters motion.InterpFilters, hasChroma bool) error {
	if err := frozenMarkInterMotionOracle(c, size, x4, y4, result, hasChroma); err != nil {
		return fmt.Errorf("mark inter motion: %w", err)
	}
	if err := frozenMarkInterFiltersOracle(c, size, x4, y4, result.References, filters); err != nil {
		return fmt.Errorf("mark inter filters: %w", err)
	}
	return nil
}

func TestDecodedInterContextMatchesOriginalCommits(t *testing.T) {
	initial := seededBlockModeContext(t, 137)
	caseIndex := 0
	for size := BlockSize(0); size < blockSizeCount; size++ {
		for _, pos := range positionsForInterContextOracle(size) {
			for _, compound := range [...]bool{false, true} {
				for _, hasChroma := range [...]bool{false, true} {
					refs := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}, Compound: compound, Unidir: hasChroma}
					if compound {
						refs.Ref[1] = ReferenceFrameAltref
					}
					result := validInterMotionForOracle(refs, caseIndex)
					filters := motion.InterpFilters{X: motion.InterpFilter(caseIndex % 4), Y: motion.InterpFilter((caseIndex / 4) % 4)}
					got, want := initial, initial
					gotErr := got.markDecodedInterContext(size, pos[0], pos[1], result, filters, hasChroma)
					wantErr := frozenDecodedInterContextOracle(&want, size, pos[0], pos[1], result, filters, hasChroma)
					if gotErr != nil || wantErr != nil || !reflect.DeepEqual(got, want) {
						t.Fatalf("combined commit differs: size=%d pos=%v compound=%t chroma=%t filters=%+v errors=%v/%v", size, pos, compound, hasChroma, filters, gotErr, wantErr)
					}
					caseIndex++
				}
			}
		}
	}
}

func TestDecodedInterContextPreservesErrorAndPartialState(t *testing.T) {
	refs := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	valid := validInterMotionForOracle(refs, 19)
	badMotion := valid
	badMotion.Mode.Mode = interModeCount
	badRefs := valid
	badRefs.References.Ref[0] = ReferenceFrameNone
	validFilters := motion.InterpFilters{X: motion.InterpEightTapSmooth, Y: motion.InterpBilinear}
	badFilters := motion.InterpFilters{X: motion.InterpFilter(255), Y: motion.InterpBilinear}
	initial := seededBlockModeContext(t, 29)
	for _, tc := range []struct {
		name    string
		size    BlockSize
		x, y    int
		result  InterMotionResult
		filters motion.InterpFilters
		partial bool
	}{
		{"filter after motion", BlockSize16x8, 2, 3, valid, badFilters, true},
		{"bad motion before filter", BlockSize16x8, 2, 3, badMotion, badFilters, false},
		{"bad refs before filter", BlockSize16x8, 2, 3, badRefs, badFilters, false},
		{"invalid size", blockSizeCount, 0, 0, valid, validFilters, false},
		{"negative x", BlockSize4x4, -1, 0, valid, validFilters, false},
		{"y past edge", BlockSize8x8, 0, MaxBlockModeSlots - 1, valid, badFilters, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, want := initial, initial
			gotErr := got.markDecodedInterContext(tc.size, tc.x, tc.y, tc.result, tc.filters, true)
			wantErr := frozenDecodedInterContextOracle(&want, tc.size, tc.x, tc.y, tc.result, tc.filters, true)
			if gotErr == nil || fmt.Sprint(gotErr) != fmt.Sprint(wantErr) || !reflect.DeepEqual(got, want) {
				t.Fatalf("combined error/state differs: errors=%v/%v", gotErr, wantErr)
			}
			if reflect.DeepEqual(got, initial) == tc.partial {
				t.Fatalf("partial update=%t was not preserved", tc.partial)
			}
		})
	}
	for _, filters := range []motion.InterpFilters{validFilters, badFilters} {
		for _, result := range []InterMotionResult{valid, badMotion} {
			got := (*BlockModeContext)(nil).markDecodedInterContext(BlockSize4x4, 0, 0, result, filters, false)
			want := frozenDecodedInterContextOracle(nil, BlockSize4x4, 0, 0, result, filters, false)
			if got == nil || fmt.Sprint(got) != fmt.Sprint(want) {
				t.Fatalf("nil combined error=%v want %v", got, want)
			}
		}
	}
}
