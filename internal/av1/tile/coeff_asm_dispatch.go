package tile

import (
	"os"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/entropy"
)

// coeffBaseLevelsKernel, coeffSignGolombKernel and mvResidualKernel gate the
// arm64 asm-spine kernels (M-D3 D3-b base-levels walk, D3-c sign/golomb
// replay and D3-e MV residual chain, spec in coeff_asm_arm64_spec.go). Static
// bool dispatch (no func pointers, per the zero-alloc rules);
// GOAV1_DISABLE_COEFF_ASM is the same-binary kill switch for the whole spine
// program: "1" (any value but the named ones) forces the pure-Go loops for
// ALL kernels, while the values "sign" and "mv" disable only the D3-c or
// D3-e kernel respectively (the incremental measurement hooks for their
// landing reviews). The differential harnesses prove every switch position
// byte-identical.
var (
	coeffAsmKillSwitch    = os.Getenv("GOAV1_DISABLE_COEFF_ASM")
	coeffBaseLevelsKernel = entropy.HasCoeffBaseLevels2D &&
		(coeffAsmKillSwitch == "" || coeffAsmKillSwitch == "sign" || coeffAsmKillSwitch == "mv")
	coeffSignGolombKernel = entropy.HasCoeffSignGolomb &&
		(coeffAsmKillSwitch == "" || coeffAsmKillSwitch == "mv")
	mvResidualKernel = entropy.HasMVResidual &&
		(coeffAsmKillSwitch == "" || coeffAsmKillSwitch == "sign")
)

// The kernel reads coeffScanHot entries as packed 8-byte words
// (pos u16 | padded u16 | lower2DOffset i8 | br2DOffset i8 | eob ctx bytes
// ignored); pin the layout at compile time.
var (
	_ [unsafe.Sizeof(coeffScanHot{}) - 8]byte
	_ [8 - unsafe.Sizeof(coeffScanHot{})]byte
	_ [unsafe.Offsetof(coeffScanHot{}.pos)]byte
	_ [unsafe.Offsetof(coeffScanHot{}.padded) - 2]byte
	_ [2 - unsafe.Offsetof(coeffScanHot{}.padded)]byte
	_ [unsafe.Offsetof(coeffScanHot{}.lower2DOffset) - 4]byte
	_ [4 - unsafe.Offsetof(coeffScanHot{}.lower2DOffset)]byte
	_ [unsafe.Offsetof(coeffScanHot{}.br2DOffset) - 5]byte
	_ [5 - unsafe.Offsetof(coeffScanHot{}.br2DOffset)]byte
)
