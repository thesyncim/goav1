package tile

import "github.com/thesyncim/goav1/internal/av1/entropy"

// partition_write.go is the forward of ReadPartition: it codes one AV1 partition
// decision into the bitstream. It mirrors ReadPartition exactly, including the
// boundary handling — when only one split direction is available the partition is
// coded as a non-adaptive binary probability gathered from the full CDF (dav1d
// decode_sb()), and when neither is available the split is forced and nothing is
// coded. The interior case codes the adaptive partition symbol.
func WritePartition(w *entropy.Writer, cdfs *PartitionCDFs, level BlockLevel, ctx int, haveHSplit, haveVSplit bool, partition Partition) error {
	if w == nil || !level.Valid() {
		return ErrInvalidDecodeState
	}
	if !haveHSplit && !haveVSplit {
		if level >= BlockLevel8x8 || partition != PartitionSplit {
			return ErrInvalidDecodeState
		}
		return nil // forced PARTITION_SPLIT, no symbol coded
	}
	if level >= BlockLevel8x8 && (!haveHSplit || !haveVSplit) {
		return ErrInvalidDecodeState
	}

	cdf, err := cdfs.CDF(level, ctx)
	if err != nil {
		return err
	}
	if haveHSplit && haveVSplit {
		if !partition.ValidForLevel(level) {
			return ErrInvalidDecodeState
		}
		w.WriteCDF(cdf, int(partition))
		return nil
	}

	values := cdf.Values()
	prob := 0
	nonSplit := PartitionH
	if haveHSplit {
		prob = gatherTopPartitionProb(values, level)
	} else {
		prob = gatherLeftPartitionProb(values, level)
		nonSplit = PartitionV
	}
	bit := 0
	switch partition {
	case PartitionSplit:
		bit = 1
	case nonSplit:
		bit = 0
	default:
		return ErrInvalidDecodeState
	}
	w.WriteBoolQ15(bit, uint32(prob))
	return nil
}
