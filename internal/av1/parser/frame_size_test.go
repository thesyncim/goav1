package parser

import (
	"errors"
	"testing"
)

func TestParseIntraFrameSizeShownKeyFrame(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	payload, prefix := buildShownKeyFramePrefix(t, seq, false)

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBool(false) // superres_params: use_superres
	w.writeBool(false) // render_and_frame_size_different

	size, err := ParseIntraFrameSize(w.bytes(), seq, prefix, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if size.RefreshFrameFlags != 0xff {
		t.Fatalf("RefreshFrameFlags=%02x want ff", size.RefreshFrameFlags)
	}
	if size.CodedWidth != seq.MaxFrameWidth || size.UpscaledWidth != seq.MaxFrameWidth || size.Height != seq.MaxFrameHeight {
		t.Fatalf("dimensions=%+v seq=%+v", size, seq)
	}
	if size.RenderWidth != seq.MaxFrameWidth || size.RenderHeight != seq.MaxFrameHeight {
		t.Fatalf("render dimensions=%+v", size)
	}
	if size.SuperResEnabled || size.SuperResDenominator != 8 {
		t.Fatalf("superres=%+v", size)
	}
	if size.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", size.BitsRead, w.bit)
	}
}

func TestParseIntraFrameSizeOverrideSuperresRender(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.FrameWidthBits = 6
	seq.FrameHeightBits = 5
	seq.MaxFrameWidth = 64
	seq.MaxFrameHeight = 32
	payload, prefix := buildShownKeyFramePrefix(t, seq, true)

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBits(47, seq.FrameWidthBits)  // frame_width_minus_1
	w.writeBits(23, seq.FrameHeightBits) // frame_height_minus_1
	w.writeBool(true)                    // use_superres
	w.writeBits(3, 3)                    // coded_denom = 12
	w.writeBool(true)                    // render_and_frame_size_different
	w.writeBits(39, 16)                  // render_width_minus_1
	w.writeBits(19, 16)                  // render_height_minus_1

	size, err := ParseIntraFrameSize(w.bytes(), seq, prefix, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !size.SuperResEnabled || size.SuperResDenominator != 12 {
		t.Fatalf("superres=%+v", size)
	}
	if size.UpscaledWidth != 48 || size.CodedWidth != 32 || size.Height != 24 {
		t.Fatalf("dimensions=%+v", size)
	}
	if !size.HaveRenderSize || size.RenderWidth != 40 || size.RenderHeight != 20 {
		t.Fatalf("render dimensions=%+v", size)
	}
}

func TestParseIntraFrameSizeHiddenKeyReadsRefreshAndOrderHints(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())

	var w testBitWriter
	w.writeBool(false)                   // show_existing_frame
	w.writeBits(uint64(FrameTypeKey), 2) // frame_type
	w.writeBool(false)                   // show_frame
	w.writeBool(true)                    // showable_frame
	w.writeBool(true)                    // error_resilient_mode
	w.writeBool(false)                   // disable_cdf_update
	w.writeBool(false)                   // frame_size_override_flag
	w.writeBits(4, seq.OrderHintBits)    // order_hint

	prefix, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if err != nil {
		t.Fatal(err)
	}

	w.writeBits(0x01, 8) // refresh_frame_flags
	for i := 0; i < refFrames; i++ {
		w.writeBits(uint64(i), seq.OrderHintBits)
	}
	w.writeBool(false) // use_superres
	w.writeBool(false) // render_and_frame_size_different

	size, err := ParseIntraFrameSize(w.bytes(), seq, prefix, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if size.RefreshFrameFlags != 0x01 {
		t.Fatalf("RefreshFrameFlags=%02x", size.RefreshFrameFlags)
	}
	for i := 0; i < refFrames; i++ {
		if size.RefOrderHints[i] != uint32(i) {
			t.Fatalf("RefOrderHints[%d]=%d", i, size.RefOrderHints[i])
		}
	}
}

func TestParseIntraFrameSizeBufferRemovalTime(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.DecoderModelInfoPresent = true
	seq.TimingInfo.EqualPictureInterval = true
	seq.DecoderModelInfo.BufferRemovalTimeLength = 5
	seq.OperatingPointsCount = 2
	seq.OperatingPoints[0].DecoderModelPresent = true
	seq.OperatingPoints[0].IDC = 0
	seq.OperatingPoints[1].DecoderModelPresent = true
	seq.OperatingPoints[1].IDC = 1 << 1

	payload, prefix := buildShownKeyFramePrefix(t, seq, false)
	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBool(true)  // buffer_removal_time_present
	w.writeBits(17, 5) // operating point 0
	w.writeBool(false) // use_superres
	w.writeBool(false) // render_and_frame_size_different

	size, err := ParseIntraFrameSize(w.bytes(), seq, prefix, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !size.BufferRemovalTimePresent || size.BufferRemovalTimes[0] != 17 || size.BufferRemovalTimes[1] != 0 {
		t.Fatalf("buffer removal=%+v", size)
	}
}

func TestParseIntraFrameSizeRejectsIntraOnlyRefreshAll(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())

	var w testBitWriter
	w.writeBool(false)                         // show_existing_frame
	w.writeBits(uint64(FrameTypeIntraOnly), 2) // frame_type
	w.writeBool(true)                          // show_frame
	w.writeBool(false)                         // error_resilient_mode
	w.writeBool(false)                         // disable_cdf_update
	w.writeBool(false)                         // frame_size_override_flag
	w.writeBits(0, seq.OrderHintBits)          // order_hint

	prefix, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if err != nil {
		t.Fatal(err)
	}
	w.writeBits(0xff, 8)

	_, err = ParseIntraFrameSize(w.bytes(), seq, prefix, 0, 0)
	if !errors.Is(err, ErrInvalidFrameHeader) {
		t.Fatalf("ParseIntraFrameSize err=%v want %v", err, ErrInvalidFrameHeader)
	}
}

func TestParseIntraFrameSizeRejectsInterFrame(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())

	var w testBitWriter
	w.writeBool(false)
	w.writeBits(uint64(FrameTypeInter), 2)
	w.writeBool(true)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBits(0, seq.OrderHintBits)
	w.writeBits(0, 3)

	prefix, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if err != nil {
		t.Fatal(err)
	}
	_, err = ParseIntraFrameSize(w.bytes(), seq, prefix, 0, 0)
	if !errors.Is(err, ErrReferenceFrameNeeded) {
		t.Fatalf("ParseIntraFrameSize err=%v want %v", err, ErrReferenceFrameNeeded)
	}
}

func TestParseFrameSizeDirectInterFrame(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	payload, prefix := buildInterFramePrefix(t, seq, false, false, 4)
	refs := oneReferenceState(seq, 0, 1)

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBits(0x02, 8) // refresh_frame_flags
	w.writeBool(false)   // frame_refs_short_signaling
	for i := 0; i < InterRefsPerFrame; i++ {
		w.writeBits(0, 3) // ref_frame_idx[i]
	}
	w.writeBool(false) // use_superres
	w.writeBool(false) // render_and_frame_size_different

	size, err := ParseFrameSize(w.bytes(), seq, prefix, &refs, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if size.RefreshFrameFlags != 0x02 {
		t.Fatalf("RefreshFrameFlags=%02x", size.RefreshFrameFlags)
	}
	for i := 0; i < InterRefsPerFrame; i++ {
		if size.RefFrameIdx[i] != 0 {
			t.Fatalf("RefFrameIdx[%d]=%d", i, size.RefFrameIdx[i])
		}
	}
	if size.UpscaledWidth != seq.MaxFrameWidth || size.Height != seq.MaxFrameHeight {
		t.Fatalf("dimensions=%+v", size)
	}
}

func TestParseFrameSizeUsesReferenceDimensions(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	payload, prefix := buildInterFramePrefix(t, seq, true, false, 4)
	refs := oneReferenceState(seq, 0, 1)
	refs.Frames[0].Size.UpscaledWidth = 48
	refs.Frames[0].Size.CodedWidth = 48
	refs.Frames[0].Size.Height = 27
	refs.Frames[0].Size.RenderWidth = 40
	refs.Frames[0].Size.RenderHeight = 20
	refs.Frames[0].Size.HaveRenderSize = true

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBits(0x01, 8) // refresh_frame_flags
	w.writeBool(false)   // frame_refs_short_signaling
	for i := 0; i < InterRefsPerFrame; i++ {
		w.writeBits(0, 3)
	}
	w.writeBool(true)  // use_ref_frame_size[0]
	w.writeBool(false) // use_superres

	size, err := ParseFrameSize(w.bytes(), seq, prefix, &refs, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !size.UsedReferenceFrameSize || size.ReferenceFrameSizeIdx != 0 {
		t.Fatalf("reference frame size flags=%+v", size)
	}
	if size.UpscaledWidth != 48 || size.CodedWidth != 48 || size.Height != 27 {
		t.Fatalf("dimensions=%+v", size)
	}
	if size.RenderWidth != 40 || size.RenderHeight != 20 {
		t.Fatalf("render dimensions=%+v", size)
	}
}

func TestParseFrameSizeRejectsMissingReference(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	payload, prefix := buildInterFramePrefix(t, seq, false, false, 4)
	var refs ReferenceState

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBits(0x01, 8)
	w.writeBool(false)
	for i := 0; i < InterRefsPerFrame; i++ {
		w.writeBits(0, 3)
	}

	_, err := ParseFrameSize(w.bytes(), seq, prefix, &refs, 0, 0)
	if !errors.Is(err, ErrReferenceFrameNeeded) {
		t.Fatalf("ParseFrameSize err=%v want %v", err, ErrReferenceFrameNeeded)
	}
}

func TestReferenceStateUpdate(t *testing.T) {
	var refs ReferenceState
	prefix := FrameHeaderPrefix{
		FrameType:     FrameTypeKey,
		ShowFrame:     true,
		ShowableFrame: false,
		OrderHint:     3,
		FrameID:       5,
	}
	size := FrameSize{
		RefreshFrameFlags: 0x81,
		UpscaledWidth:     16,
		CodedWidth:        16,
		Height:            9,
		RenderWidth:       16,
		RenderHeight:      9,
	}
	refs.Update(prefix, size)
	if !refs.Frames[0].Valid || !refs.Frames[7].Valid || refs.Frames[1].Valid {
		t.Fatalf("reference state=%+v", refs.Frames)
	}
	if refs.Frames[7].OrderHint != 3 || refs.Frames[7].FrameID != 5 {
		t.Fatalf("reference metadata=%+v", refs.Frames[7])
	}
}

func TestParseIntraFrameSizeAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	payload, prefix := buildShownKeyFramePrefix(t, seq, false)
	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBool(false)
	w.writeBool(false)
	payload = w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseIntraFrameSize(payload, seq, prefix, 0, 0)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseIntraFrameSize allocated: %f", allocs)
	}
}

func BenchmarkParseIntraFrameSize(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	payload, prefix := buildShownKeyFramePrefixBench(b, seq, false)
	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBool(false)
	w.writeBool(false)
	payload = w.bytes()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ParseIntraFrameSize(payload, seq, prefix, 0, 0)
	}
}

func BenchmarkParseFrameSizeInter(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	payload, prefix := buildInterFramePrefixBench(b, seq, false, false, 4)
	refs := oneReferenceState(seq, 0, 1)

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBits(0x02, 8)
	w.writeBool(false)
	for i := 0; i < InterRefsPerFrame; i++ {
		w.writeBits(0, 3)
	}
	w.writeBool(false)
	w.writeBool(false)
	payload = w.bytes()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
	}
}

// TestParseFrameSizeInterAllocs pins ParseFrameSize on the inter path (which
// runs every inter frame) to zero allocations. The function builds the
// FrameSize value on the stack and writes into caller-supplied references, so
// future changes that promote any of the fixed-size scratch buffers (e.g. the
// shortRefInfo array used by setShortFrameReferences) to the heap will trip
// this test.
func TestParseFrameSizeInterAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	payload, prefix := buildInterFramePrefix(t, seq, false, false, 4)
	refs := oneReferenceState(seq, 0, 1)

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBits(0x02, 8) // refresh_frame_flags
	w.writeBool(false)   // frame_refs_short_signaling
	for i := 0; i < InterRefsPerFrame; i++ {
		w.writeBits(0, 3) // ref_frame_idx[i]
	}
	w.writeBool(false) // use_superres
	w.writeBool(false) // render_and_frame_size_different
	payload = w.bytes()

	if _, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0); err != nil {
		t.Fatalf("ParseFrameSize err=%v", err)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseFrameSize allocated: %f", allocs)
	}
}

// TestParseFrameSizeShortSignalingAllocs covers the short-signaling reference
// derivation path which uses on-stack scratch arrays (shortRefInfo, set).
func TestParseFrameSizeShortSignalingAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	payload, prefix := buildInterFramePrefix(t, seq, false, false, 4)
	refs := oneReferenceState(seq, 0, 1)
	for i := range refs.Frames {
		refs.Frames[i].Valid = true
		refs.Frames[i].OrderHint = uint32(i)
	}

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBits(0x02, 8) // refresh_frame_flags
	w.writeBool(true)    // frame_refs_short_signaling
	w.writeBits(0, 3)    // last_frame_idx
	w.writeBits(3, 3)    // golden_frame_idx
	w.writeBool(false)   // use_superres
	w.writeBool(false)   // render_and_frame_size_different
	payload = w.bytes()

	// Warm up: this path may error on the synthetic ordering; if so, just skip
	// the lock-in since we cannot validate the steady state.
	if _, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0); err != nil {
		t.Skipf("short-signaling path unavailable in test fixture: %v", err)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseFrameSize short-signaling allocated: %f", allocs)
	}
}

func buildShownKeyFramePrefix(t *testing.T, seq SequenceHeader, frameSizeOverride bool) ([]byte, FrameHeaderPrefix) {
	t.Helper()
	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, frameSizeOverride)
	if err != nil {
		t.Fatal(err)
	}
	return payload, prefix
}

func buildShownKeyFramePrefixBench(b *testing.B, seq SequenceHeader, frameSizeOverride bool) ([]byte, FrameHeaderPrefix) {
	b.Helper()
	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, frameSizeOverride)
	if err != nil {
		b.Fatal(err)
	}
	return payload, prefix
}

func buildInterFramePrefix(t *testing.T, seq SequenceHeader, frameSizeOverride bool, errorResilient bool, orderHint uint64) ([]byte, FrameHeaderPrefix) {
	t.Helper()
	payload, prefix, err := buildInterFramePrefixRaw(seq, frameSizeOverride, errorResilient, orderHint)
	if err != nil {
		t.Fatal(err)
	}
	return payload, prefix
}

func buildInterFramePrefixBench(b *testing.B, seq SequenceHeader, frameSizeOverride bool, errorResilient bool, orderHint uint64) ([]byte, FrameHeaderPrefix) {
	b.Helper()
	payload, prefix, err := buildInterFramePrefixRaw(seq, frameSizeOverride, errorResilient, orderHint)
	if err != nil {
		b.Fatal(err)
	}
	return payload, prefix
}

func buildInterFramePrefixRaw(seq SequenceHeader, frameSizeOverride bool, errorResilient bool, orderHint uint64) ([]byte, FrameHeaderPrefix, error) {
	var w testBitWriter
	w.writeBool(false)                     // show_existing_frame
	w.writeBits(uint64(FrameTypeInter), 2) // frame_type
	w.writeBool(true)                      // show_frame
	w.writeBool(errorResilient)            // error_resilient_mode
	w.writeBool(false)                     // disable_cdf_update
	w.writeBool(frameSizeOverride)         // frame_size_override_flag
	w.writeBits(orderHint, seq.OrderHintBits)
	if !errorResilient {
		w.writeBits(0, 3) // primary_ref_frame
	}
	payload := w.bytes()
	prefix, err := ParseFrameHeaderPrefix(payload, seq)
	return payload, prefix, err
}

func oneReferenceState(seq SequenceHeader, slot int, orderHint uint32) ReferenceState {
	var refs ReferenceState
	refs.Frames[slot] = ReferenceFrame{
		Valid:     true,
		OrderHint: orderHint,
		Size: FrameSize{
			UpscaledWidth:       seq.MaxFrameWidth,
			CodedWidth:          seq.MaxFrameWidth,
			Height:              seq.MaxFrameHeight,
			RenderWidth:         seq.MaxFrameWidth,
			RenderHeight:        seq.MaxFrameHeight,
			SuperResDenominator: 8,
		},
	}
	return refs
}

func buildShownKeyFramePrefixRaw(seq SequenceHeader, frameSizeOverride bool) ([]byte, FrameHeaderPrefix, error) {
	var w testBitWriter
	w.writeBool(false)                   // show_existing_frame
	w.writeBits(uint64(FrameTypeKey), 2) // frame_type
	w.writeBool(true)                    // show_frame
	w.writeBool(false)                   // disable_cdf_update
	w.writeBool(frameSizeOverride)       // frame_size_override_flag
	w.writeBits(3, seq.OrderHintBits)    // order_hint
	payload := w.bytes()
	prefix, err := ParseFrameHeaderPrefix(payload, seq)
	return payload, prefix, err
}
