package work

// TilePlan is the caller-buffer work description derived from one tile group
// event.
type TilePlan struct {
	SpanCount  int
	JobCount   int
	BatchCount int
}

// FramePlan is the caller-buffer work description for one frame-begin event.
// Frame-header events have no tile work yet; frame events may carry an
// implicit final tile group.
type FramePlan struct {
	Surface        int
	ReferenceCount int
	Tile           TilePlan
}

// FrameTilePlan binds a tile-group work plan to the output surface and
// resolved reference count chosen when the frame began.
type FrameTilePlan struct {
	Surface        int
	ReferenceCount int
	Tile           TilePlan
}

// ShowExistingFramePlan is the caller-buffer work result for a
// show-existing-frame event. Surface is the frame-pool slot to output.
type ShowExistingFramePlan struct {
	Surface          int
	ReleaseCount     int
	DroppedFrameWork bool
}

// FrameStepKind identifies a decoder frame-work action.
type FrameStepKind uint8

const (
	// FrameStepIgnored means the event did not affect frame work.
	FrameStepIgnored FrameStepKind = iota
	// FrameStepDropped means active incomplete frame work was aborted.
	FrameStepDropped
	// FrameStepBegin means a new output surface was acquired.
	FrameStepBegin
	// FrameStepTile means continuation tile work was planned.
	FrameStepTile
	// FrameStepShowExisting means an existing reference surface is output.
	FrameStepShowExisting
)

// FrameStep is the caller-buffer result of applying one decoder event to
// frame-work state. The active frame should be finished separately after the
// caller executes any tile work reported by Begin or Tile.
type FrameStep struct {
	Kind             FrameStepKind
	DroppedFrameWork bool

	Begin        FramePlan
	Tile         FrameTilePlan
	ShowExisting ShowExistingFramePlan
}

// FrameStepResult reports the side effects completed by a frame-work run step.
type FrameStepResult struct {
	ExecutedTileWork bool
	CompletedFrame   bool
	ReleaseCount     int
}
