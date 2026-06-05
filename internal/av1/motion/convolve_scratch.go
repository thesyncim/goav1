package motion

// ConvolveScratch carries reusable temporary storage for translational 2D
// inter-prediction convolution. It keeps the large int16 intermediate block out
// of hot call frames on framework decode paths.
type ConvolveScratch struct {
	im [(maxBlockSize + filterTaps - 1) * maxBlockSize]int16
}
