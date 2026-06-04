package threading

import "github.com/thesyncim/goav1/internal/av1/tile"

// FrameWorkRestorationFrameBuffers binds the caller-owned loop-restoration
// record and stripe-boundary storage for one frame.
type FrameWorkRestorationFrameBuffers struct {
	Plan       tile.RestorationFramePlan
	Records    [3][]tile.RestorationUnitRecord
	Boundaries [3]tile.RestorationStripeBoundaries
}

// BindRestorationFrameBuffers validates and partitions caller-owned storage for
// decoded restoration records and saved stripe boundaries.
func (b *FrameWorkBatch) BindRestorationFrameBuffers(recordBacking []tile.RestorationUnitRecord, above []uint16, below []uint16) (FrameWorkRestorationFrameBuffers, error) {
	plan, err := b.RestorationFramePlan()
	if err != nil {
		return FrameWorkRestorationFrameBuffers{}, err
	}
	records, err := tile.BindRestorationFrameRecordBuffers(plan, recordBacking)
	if err != nil {
		return FrameWorkRestorationFrameBuffers{}, err
	}
	boundaries, err := tile.BindRestorationFrameBoundaryBuffers(plan, above, below)
	if err != nil {
		return FrameWorkRestorationFrameBuffers{}, err
	}
	return FrameWorkRestorationFrameBuffers{
		Plan:       plan,
		Records:    records,
		Boundaries: boundaries,
	}, nil
}

// ResetRecords initializes every active plane's frame-wide unit-info-equivalent
// record buffer to RESTORE_NONE records with precomputed geometry.
func (b FrameWorkRestorationFrameBuffers) ResetRecords() error {
	switch b.Plan.Planes {
	case 1, 3:
	default:
		return tile.ErrInvalidPlan
	}
	for plane := 0; plane < int(b.Plan.Planes); plane++ {
		if b.Plan.UnitRecords[plane] == 0 {
			if len(b.Records[plane]) != 0 {
				return tile.ErrInvalidPlan
			}
			continue
		}
		if err := tile.ResetRestorationPlaneRecords(b.Plan.Grids[plane], b.Records[plane]); err != nil {
			return err
		}
	}
	return nil
}

// ReadRestorationUnitsForSuperblock decodes every loop-restoration unit whose
// top-left corner belongs to visit and writes it into the frame-wide record
// buffers carried by req.
func (b *FrameWorkBatch) ReadRestorationUnitsForSuperblock(index int, state *tile.DecodeState, cdfs tile.RestorationCDFs, req *FrameWorkTileRestorationRequest, visit tile.BlockLoopSuperblockVisit) (int, error) {
	if state == nil || req == nil {
		return 0, ErrInvalidBatch
	}
	region, err := b.JobRegion(index)
	if err != nil {
		return 0, err
	}
	if visit.SBSizeMIB != b.Sequence.SBSizeMIB ||
		visit.MICol < uint32(region.MIColStart) ||
		visit.MIRow < uint32(region.MIRowStart) ||
		visit.MICol >= uint32(region.MIColEnd) ||
		visit.MIRow >= uint32(region.MIRowEnd) {
		return 0, ErrInvalidBatch
	}
	if req.Buffers.Plan.Planes != 1 && req.Buffers.Plan.Planes != 3 {
		return 0, tile.ErrInvalidPlan
	}
	total := 0
	for plane := 0; plane < int(req.Buffers.Plan.Planes); plane++ {
		grid := req.Buffers.Plan.Grids[plane]
		if req.Buffers.Plan.UnitRecords[plane] == 0 {
			if len(req.Buffers.Records[plane]) != 0 {
				return 0, tile.ErrInvalidPlan
			}
			continue
		}
		n, err := state.ReadRestorationUnitsForSuperblockInto(grid, visit.MICol, visit.MIRow, visit.SBSizeMIB, req.Buffers.Records[plane], &req.References[plane], cdfs)
		if err != nil {
			return total, err
		}
		total += n
	}
	return total, nil
}
