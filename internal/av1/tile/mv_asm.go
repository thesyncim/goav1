package tile

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/entropy"
)

// The M-D3 D3-e motion-vector residual kernel (entropy.Cursor.MVResidual,
// spec in coeff_asm_arm64_spec.go §7) hardcodes the MVCDFs field offsets as
// distances from the struct base; pin the layout at compile time so any
// change breaks the build instead of the decode.
var (
	_ [unsafe.Sizeof(entropy.CDF{}) - 36]byte
	_ [36 - unsafe.Sizeof(entropy.CDF{})]byte
	_ [unsafe.Offsetof(MVCDFs{}.Joint)]byte
	_ [unsafe.Offsetof(MVCDFs{}.Components) - 36]byte
	_ [36 - unsafe.Offsetof(MVCDFs{}.Components)]byte
	_ [unsafe.Sizeof(MVComponentCDFs{}) - 648]byte
	_ [648 - unsafe.Sizeof(MVComponentCDFs{})]byte
	_ [unsafe.Offsetof(MVComponentCDFs{}.Classes)]byte
	_ [unsafe.Offsetof(MVComponentCDFs{}.Class0FP) - 36]byte
	_ [36 - unsafe.Offsetof(MVComponentCDFs{}.Class0FP)]byte
	_ [unsafe.Offsetof(MVComponentCDFs{}.FP) - 108]byte
	_ [108 - unsafe.Offsetof(MVComponentCDFs{}.FP)]byte
	_ [unsafe.Offsetof(MVComponentCDFs{}.Sign) - 144]byte
	_ [144 - unsafe.Offsetof(MVComponentCDFs{}.Sign)]byte
	_ [unsafe.Offsetof(MVComponentCDFs{}.Class0HP) - 180]byte
	_ [180 - unsafe.Offsetof(MVComponentCDFs{}.Class0HP)]byte
	_ [unsafe.Offsetof(MVComponentCDFs{}.HP) - 216]byte
	_ [216 - unsafe.Offsetof(MVComponentCDFs{}.HP)]byte
	_ [unsafe.Offsetof(MVComponentCDFs{}.Class0) - 252]byte
	_ [252 - unsafe.Offsetof(MVComponentCDFs{}.Class0)]byte
	_ [unsafe.Offsetof(MVComponentCDFs{}.Bits) - 288]byte
	_ [288 - unsafe.Offsetof(MVComponentCDFs{}.Bits)]byte
)

// mvCDFsKernelShape reports whether every row the kernel may touch has the
// canonical nmv_context shape. The pure-Go chain validates rows as it reaches
// them; the kernel validates them all upfront, so a malformed set falls back
// to the pure path, which reproduces the exact mid-chain error byte-for-byte.
func mvCDFsKernelShape(cdfs *MVCDFs) bool {
	if cdfs.Joint.Symbols() != MVJoints {
		return false
	}
	for i := range cdfs.Components {
		c := &cdfs.Components[i]
		if c.Sign.Symbols() != 2 ||
			c.Classes.Symbols() != MVClasses ||
			c.Class0.Symbols() != MVClass0Size ||
			c.FP.Symbols() != MVFPSize ||
			c.Class0FP[0].Symbols() != MVFPSize ||
			c.Class0FP[1].Symbols() != MVFPSize ||
			c.Class0HP.Symbols() != 2 ||
			c.HP.Symbols() != 2 {
			return false
		}
		for j := range c.Bits {
			if c.Bits[j].Symbols() != 2 {
				return false
			}
		}
	}
	return true
}

// unpackMVKernelComponent expands one packed kernel component (see
// entropy.Cursor.MVResidual) into the MVComponentResult the pure-Go chain
// produces.
func unpackMVKernelComponent(packed int64) MVComponentResult {
	u := uint64(packed)
	class := uint8(u>>26) & 0xf
	return MVComponentResult{
		Diff:          int16(uint16(u)),
		IntegerPart:   uint16(u>>16) & 0x3ff,
		Class:         class,
		Fraction:      uint8(u>>30) & 3,
		HighPrecision: uint8(u>>32) & 1,
		Sign:          u>>33&1 != 0,
		Class0:        class == 0,
	}
}

// readMotionVectorKernel decodes one MV residual through the arm64 kernel.
// The caller has validated the CDF shape (mvCDFsKernelShape) and the
// precision, so the chain cannot fail.
func readMotionVectorKernel(reader *entropy.Cursor, cdfs *MVCDFs, precision MVSubpelPrecision) MVResidualResult {
	joint, comp0, comp1 := reader.MVResidual(unsafe.Pointer(cdfs), int64(precision))
	result := MVResidualResult{Joint: MVJoint(joint), Precision: precision}
	if result.Joint.HasVertical() {
		comp := unpackMVKernelComponent(comp0)
		result.Diff.Row = comp.Diff
		result.Components[mvComponentRow] = comp
		result.ComponentValid[mvComponentRow] = true
	}
	if result.Joint.HasHorizontal() {
		comp := unpackMVKernelComponent(comp1)
		result.Diff.Col = comp.Diff
		result.Components[mvComponentCol] = comp
		result.ComponentValid[mvComponentCol] = true
	}
	return result
}
