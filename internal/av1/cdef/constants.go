package cdef

const (
	BlockSize         = 64
	MaxSuperblockSize = 128
	NBlocks           = MaxSuperblockSize / 8

	VerticalBorder   = 2
	HorizontalBorder = 8
	BStride          = MaxSuperblockSize + 2*HorizontalBorder
	VeryLarge        = 0x4000
	InputBufferSize  = BStride * (MaxSuperblockSize + 2*VerticalBorder)
)
