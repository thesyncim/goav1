package tile

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

// The oracle in motion_field_test.go keeps the old per-entry traversal and
// independently rounds its projections. Runs deliberately cross 8-cell windows,
// stop at validity/reference/vector changes, and end before poisoned padding.
func TestTemporalMotionFieldProjectionRunsMatchScalar(t *testing.T) {
	const rows, cols, sourceStride, destinationStride = 19, 41, 47, 49
	rng := rand.New(rand.NewSource(0x72854c))
	vectors := []motion.Vector{{}, {Row: 64, Col: 64}, {Row: -64, Col: -64}, {Row: 65, Col: -65}, {Row: -1, Col: 1}, {Col: 512}, {Col: -512}, {Row: 512, Col: 768}, {Row: -512, Col: -768}, {Row: 32767, Col: -32768}}
	for _, runLength := range []int{1, 2, 3, 7, 8, 9, 15, 16, 17, cols} {
		for trial := 0; trial < 12; trial++ {
			entries := make([]ReferenceMVEntry, rows*sourceStride+3)
			for i := range entries {
				entries[i] = ReferenceMVEntry{Ref: ReferenceFrameLast, MV: motion.Vector{Row: 123, Col: -456}, Valid: true}
			}
			for row := 0; row < rows; row++ {
				for col := 0; col < cols; {
					entry := ReferenceMVEntry{Ref: ReferenceFrame(1 + rng.Intn(int(referenceFrameCount))), MV: vectors[rng.Intn(len(vectors))], Valid: rng.Intn(5) != 0}
					for end := min(col+runLength, cols); col < end; col++ {
						entries[row*sourceStride+col] = entry
					}
				}
			}
			before := slices.Clone(entries)
			start := ReferenceMVFrame{Rows: rows, Cols: cols, Stride: sourceStride, Entries: entries}
			seed := make([]TemporalMotionEntry, rows*destinationStride+3)
			for i := range seed {
				seed[i] = TemporalMotionEntry{MV: motion.Vector{Row: int16(i + 1), Col: int16(-i - 1)}, RefFrameOffset: uint8(i), Valid: i&1 == 0}
			}
			got := TemporalMotionField{Rows: rows, Cols: cols, Stride: destinationStride, Entries: slices.Clone(seed)}
			want := TemporalMotionField{Rows: rows, Cols: cols, Stride: destinationStride, Entries: slices.Clone(seed)}
			var hints [referenceFrameCount]uint8
			for i := range hints {
				hints[i] = uint8((i*7 + trial) & 63)
			}
			req := TemporalMotionProjectionRequest{StartFrame: &start, OrderHintBits: 6, StartOrderHint: 32, CurrentOrderHint: uint8(trial * 5), StartRefOrderHints: hints, Backward: trial&1 != 0}
			gotApplied, gotErr := got.ProjectReferenceFrame(req)
			wantApplied, wantErr := projectTemporalMotionFieldScalarReference(&want, req)
			if gotApplied != wantApplied || gotErr != wantErr || !slices.Equal(got.Entries, want.Entries) {
				t.Fatalf("run=%d trial=%d: projection/result/padding differs: got applied=%v err=%v want applied=%v err=%v", runLength, trial, gotApplied, gotErr, wantApplied, wantErr)
			}
			if !slices.Equal(start.Entries, before) {
				t.Fatalf("run=%d trial=%d: source entries changed", runLength, trial)
			}
		}
	}
}

func TestTemporalMotionFieldProjectionRunsKeepCollisionAndValidityBoundaries(t *testing.T) {
	const rows, cols = 9, 25
	entries := make([]ReferenceMVEntry, rows*cols)
	for col := 0; col < 12; col++ {
		entries[4*cols+col] = ReferenceMVEntry{Ref: ReferenceFrameLast, MV: motion.Vector{Col: 64}, Valid: true}
	}
	for col := 12; col < cols; col++ {
		entries[4*cols+col] = ReferenceMVEntry{Ref: ReferenceFrameLast, MV: motion.Vector{Col: -64}, Valid: true}
	}
	// Both changes split a run even though the adjacent motion vectors match.
	entries[4*cols+14].Valid = false
	entries[4*cols+16].Ref = ReferenceFrameGolden
	start := ReferenceMVFrame{Rows: rows, Cols: cols, Stride: cols, Entries: entries}
	got := TemporalMotionField{Rows: rows, Cols: cols, Stride: cols, Entries: make([]TemporalMotionEntry, rows*cols)}
	want := TemporalMotionField{Rows: rows, Cols: cols, Stride: cols, Entries: make([]TemporalMotionEntry, rows*cols)}
	req := TemporalMotionProjectionRequest{StartFrame: &start, OrderHintBits: 5, StartOrderHint: 4, CurrentOrderHint: 0, StartRefOrderHints: [referenceFrameCount]uint8{ReferenceFrameLast: 0, ReferenceFrameGolden: 2}}
	_, gotErr := got.ProjectReferenceFrame(req)
	_, wantErr := projectTemporalMotionFieldScalarReference(&want, req)
	if gotErr != wantErr || !slices.Equal(got.Entries, want.Entries) {
		t.Fatalf("run collision/validity boundary mismatch: got err=%v want err=%v", gotErr, wantErr)
	}
	lastWriter := TemporalMotionEntry{MV: motion.Vector{Col: -64}, RefFrameOffset: 4, Valid: true}
	if got.Entries[4*cols+11] != lastWriter || got.Entries[4*cols+12] != lastWriter {
		t.Fatal("later run did not overwrite both overlapping destinations")
	}
	if got.Entries[4*cols+13].Valid {
		t.Fatal("invalid source entry inside a run produced a destination")
	}
}
