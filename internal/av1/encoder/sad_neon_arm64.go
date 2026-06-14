//go:build arm64 && !purego

package encoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// sad8x8NEONCtx carries the kernel arguments; the assembly writes the sum into
// Sum. Field offsets are mirrored by #define in sad_neon_arm64.s.
type sad8x8NEONCtx struct {
	Src    unsafe.Pointer
	Ref    unsafe.Pointer
	Stride int64
	Sum    int64
}

//go:noescape
func sad8x8NEONAsm(ctx *sad8x8NEONCtx)

// sad8x8NEON computes the full 8x8 SAD with NEON widening absolute-difference
// accumulation. The limit hint is ignored: the vector kernel finishes the
// block faster than the scalar early exit can help.
func sad8x8NEON(src, ref []byte, stride int, limit int) int {
	ctx := sad8x8NEONCtx{
		Src:    unsafe.Pointer(&src[0]),
		Ref:    unsafe.Pointer(&ref[0]),
		Stride: int64(stride),
	}
	sad8x8NEONAsm(&ctx)
	return int(ctx.Sum)
}

//go:noescape
func sad16x16NEONAsm(ctx *sad8x8NEONCtx)

//go:noescape
func sad16x16DotProdAsm(ctx *sad8x8NEONCtx)

var useDotProdSAD = cpu.Detected.DOTPROD

// sad16x16NEON computes the full 16x16 SAD with paired widening
// absolute-difference accumulation over 16-byte rows.
func sad16x16NEON(src, ref []byte, stride int) int {
	ctx := sad8x8NEONCtx{
		Src:    unsafe.Pointer(&src[0]),
		Ref:    unsafe.Pointer(&ref[0]),
		Stride: int64(stride),
	}
	sad16x16NEONAsm(&ctx)
	return int(ctx.Sum)
}

// sad16x16DotProd computes the full 16x16 SAD with SVT-style UABD+UDOT.
func sad16x16DotProd(src, ref []byte, stride int) int {
	ctx := sad8x8NEONCtx{
		Src:    unsafe.Pointer(&src[0]),
		Ref:    unsafe.Pointer(&ref[0]),
		Stride: int64(stride),
	}
	sad16x16DotProdAsm(&ctx)
	return int(ctx.Sum)
}

//go:noescape
func sad32x32NEONAsm(ctx *sad8x8NEONCtx)

//go:noescape
func sad32x32DotProdAsm(ctx *sad8x8NEONCtx)

// sad32x32NEON computes the full 32x32 SAD in one assembly call.
func sad32x32NEON(src, ref []byte, stride int) int {
	ctx := sad8x8NEONCtx{
		Src:    unsafe.Pointer(&src[0]),
		Ref:    unsafe.Pointer(&ref[0]),
		Stride: int64(stride),
	}
	sad32x32NEONAsm(&ctx)
	return int(ctx.Sum)
}

// sad32x32DotProd computes the full 32x32 SAD with SVT-style UABD+UDOT.
func sad32x32DotProd(src, ref []byte, stride int) int {
	ctx := sad8x8NEONCtx{
		Src:    unsafe.Pointer(&src[0]),
		Ref:    unsafe.Pointer(&ref[0]),
		Stride: int64(stride),
	}
	sad32x32DotProdAsm(&ctx)
	return int(ctx.Sum)
}

// sad8x8x4Step4NEONCtx carries four horizontal 8x8 SAD outputs for reference
// origins separated by four bytes. Field offsets are mirrored in assembly.
type sad8x8x4Step4NEONCtx struct {
	Src    unsafe.Pointer
	Ref    unsafe.Pointer
	Stride int64
	Sum0   int64
	Sum1   int64
	Sum2   int64
	Sum3   int64
}

//go:noescape
func sad8x8x4Step4NEONAsm(ctx *sad8x8x4Step4NEONCtx)

//go:noescape
func sad16x16x4Step4NEONAsm(ctx *sad8x8x4Step4NEONCtx)

//go:noescape
func sad16x16x4Step4DotProdAsm(ctx *sad8x8x4Step4NEONCtx)

//go:noescape
func sad32x32x4Step4NEONAsm(ctx *sad8x8x4Step4NEONCtx)

//go:noescape
func sad32x32x4Step4DotProdAsm(ctx *sad8x8x4Step4NEONCtx)

//go:noescape
func sad32x32x4NEONAsm(ctx *sad8x8x4NEONCtx)

func sad8x8x4Step4NEON(src, ref []byte, stride int) (int, int, int, int) {
	ctx := sad8x8x4Step4NEONCtx{
		Src:    unsafe.Pointer(&src[0]),
		Ref:    unsafe.Pointer(&ref[0]),
		Stride: int64(stride),
	}
	sad8x8x4Step4NEONAsm(&ctx)
	return int(ctx.Sum0), int(ctx.Sum1), int(ctx.Sum2), int(ctx.Sum3)
}

// sad8x8x4NEONCtx carries four arbitrary 8x8 reference origins for one shared
// source block. Field offsets are mirrored in assembly.
type sad8x8x4NEONCtx struct {
	Src    unsafe.Pointer
	Ref0   unsafe.Pointer
	Ref1   unsafe.Pointer
	Ref2   unsafe.Pointer
	Ref3   unsafe.Pointer
	Stride int64
	Sum0   int64
	Sum1   int64
	Sum2   int64
	Sum3   int64
}

//go:noescape
func sad8x8x4NEONAsm(ctx *sad8x8x4NEONCtx)

//go:noescape
func sad16x16x4NEONAsm(ctx *sad8x8x4NEONCtx)

//go:noescape
func sad16x16x4DotProdAsm(ctx *sad8x8x4NEONCtx)

//go:noescape
func sad32x32x4DotProdAsm(ctx *sad8x8x4NEONCtx)

func sad8x8x4NEON(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	ctx := sad8x8x4NEONCtx{
		Src:    unsafe.Pointer(&src[0]),
		Ref0:   unsafe.Pointer(&ref0[0]),
		Ref1:   unsafe.Pointer(&ref1[0]),
		Ref2:   unsafe.Pointer(&ref2[0]),
		Ref3:   unsafe.Pointer(&ref3[0]),
		Stride: int64(stride),
	}
	sad8x8x4NEONAsm(&ctx)
	return int(ctx.Sum0), int(ctx.Sum1), int(ctx.Sum2), int(ctx.Sum3)
}

func sad16x16x4NEON(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	ctx := sad8x8x4NEONCtx{
		Src:    unsafe.Pointer(&src[0]),
		Ref0:   unsafe.Pointer(&ref0[0]),
		Ref1:   unsafe.Pointer(&ref1[0]),
		Ref2:   unsafe.Pointer(&ref2[0]),
		Ref3:   unsafe.Pointer(&ref3[0]),
		Stride: int64(stride),
	}
	sad16x16x4NEONAsm(&ctx)
	return int(ctx.Sum0), int(ctx.Sum1), int(ctx.Sum2), int(ctx.Sum3)
}

func sad16x16x4DotProd(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	ctx := sad8x8x4NEONCtx{
		Src:    unsafe.Pointer(&src[0]),
		Ref0:   unsafe.Pointer(&ref0[0]),
		Ref1:   unsafe.Pointer(&ref1[0]),
		Ref2:   unsafe.Pointer(&ref2[0]),
		Ref3:   unsafe.Pointer(&ref3[0]),
		Stride: int64(stride),
	}
	sad16x16x4DotProdAsm(&ctx)
	return int(ctx.Sum0), int(ctx.Sum1), int(ctx.Sum2), int(ctx.Sum3)
}

func sad32x32x4NEON(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	ctx := sad8x8x4NEONCtx{
		Src:    unsafe.Pointer(&src[0]),
		Ref0:   unsafe.Pointer(&ref0[0]),
		Ref1:   unsafe.Pointer(&ref1[0]),
		Ref2:   unsafe.Pointer(&ref2[0]),
		Ref3:   unsafe.Pointer(&ref3[0]),
		Stride: int64(stride),
	}
	sad32x32x4NEONAsm(&ctx)
	return int(ctx.Sum0), int(ctx.Sum1), int(ctx.Sum2), int(ctx.Sum3)
}

func sad32x32x4DotProd(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	ctx := sad8x8x4NEONCtx{
		Src:    unsafe.Pointer(&src[0]),
		Ref0:   unsafe.Pointer(&ref0[0]),
		Ref1:   unsafe.Pointer(&ref1[0]),
		Ref2:   unsafe.Pointer(&ref2[0]),
		Ref3:   unsafe.Pointer(&ref3[0]),
		Stride: int64(stride),
	}
	sad32x32x4DotProdAsm(&ctx)
	return int(ctx.Sum0), int(ctx.Sum1), int(ctx.Sum2), int(ctx.Sum3)
}

func sad16x16x4Step4NEON(src, ref []byte, stride int) (int, int, int, int) {
	ctx := sad8x8x4Step4NEONCtx{
		Src:    unsafe.Pointer(&src[0]),
		Ref:    unsafe.Pointer(&ref[0]),
		Stride: int64(stride),
	}
	sad16x16x4Step4NEONAsm(&ctx)
	return int(ctx.Sum0), int(ctx.Sum1), int(ctx.Sum2), int(ctx.Sum3)
}

func sad16x16x4Step4DotProd(src, ref []byte, stride int) (int, int, int, int) {
	ctx := sad8x8x4Step4NEONCtx{
		Src:    unsafe.Pointer(&src[0]),
		Ref:    unsafe.Pointer(&ref[0]),
		Stride: int64(stride),
	}
	sad16x16x4Step4DotProdAsm(&ctx)
	return int(ctx.Sum0), int(ctx.Sum1), int(ctx.Sum2), int(ctx.Sum3)
}

func sad32x32x4Step4NEON(src, ref []byte, stride int) (int, int, int, int) {
	ctx := sad8x8x4Step4NEONCtx{
		Src:    unsafe.Pointer(&src[0]),
		Ref:    unsafe.Pointer(&ref[0]),
		Stride: int64(stride),
	}
	sad32x32x4Step4NEONAsm(&ctx)
	return int(ctx.Sum0), int(ctx.Sum1), int(ctx.Sum2), int(ctx.Sum3)
}

func sad32x32x4Step4DotProd(src, ref []byte, stride int) (int, int, int, int) {
	ctx := sad8x8x4Step4NEONCtx{
		Src:    unsafe.Pointer(&src[0]),
		Ref:    unsafe.Pointer(&ref[0]),
		Stride: int64(stride),
	}
	sad32x32x4Step4DotProdAsm(&ctx)
	return int(ctx.Sum0), int(ctx.Sum1), int(ctx.Sum2), int(ctx.Sum3)
}

// sad8x8DualNEONCtx carries the two-stride kernel arguments.
type sad8x8DualNEONCtx struct {
	Src       unsafe.Pointer
	Ref       unsafe.Pointer
	SrcStride int64
	RefStride int64
	Sum       int64
}

//go:noescape
func sad8x8DualNEONAsm(ctx *sad8x8DualNEONCtx)

//go:noescape
func sad16x16DualNEONAsm(ctx *sad8x8DualNEONCtx)

//go:noescape
func sad32x32DualNEONAsm(ctx *sad8x8DualNEONCtx)

// sad8x8DualNEON is the NEON 8x8 SAD with independent source and reference
// strides.
func sad8x8DualNEON(src []byte, srcStride int, ref []byte, refStride int) int {
	ctx := sad8x8DualNEONCtx{
		Src:       unsafe.Pointer(&src[0]),
		Ref:       unsafe.Pointer(&ref[0]),
		SrcStride: int64(srcStride),
		RefStride: int64(refStride),
	}
	sad8x8DualNEONAsm(&ctx)
	return int(ctx.Sum)
}

// sad16x16DualNEON is the NEON 16x16 SAD with independent source and
// reference strides.
func sad16x16DualNEON(src []byte, srcStride int, ref []byte, refStride int) int {
	ctx := sad8x8DualNEONCtx{
		Src:       unsafe.Pointer(&src[0]),
		Ref:       unsafe.Pointer(&ref[0]),
		SrcStride: int64(srcStride),
		RefStride: int64(refStride),
	}
	sad16x16DualNEONAsm(&ctx)
	return int(ctx.Sum)
}

// sad32x32DualNEON is the NEON 32x32 SAD with independent source and
// reference strides.
func sad32x32DualNEON(src []byte, srcStride int, ref []byte, refStride int) int {
	ctx := sad8x8DualNEONCtx{
		Src:       unsafe.Pointer(&src[0]),
		Ref:       unsafe.Pointer(&ref[0]),
		SrcStride: int64(srcStride),
		RefStride: int64(refStride),
	}
	sad32x32DualNEONAsm(&ctx)
	return int(ctx.Sum)
}

type sad8x8CompoundAvgNEONCtx struct {
	Src        unsafe.Pointer
	Ref0       unsafe.Pointer
	Ref1       unsafe.Pointer
	SrcStride  int64
	Ref0Stride int64
	Ref1Stride int64
	Sum        int64
}

//go:noescape
func sad8x8CompoundAvgNEONAsm(ctx *sad8x8CompoundAvgNEONCtx)

// sad8x8CompoundAvgBlockNEON computes SAD(src, urhadd(ref0, ref1)) over one
// 8x8 block with independent source/reference strides.
func sad8x8CompoundAvgBlockNEON(src []byte, srcStride int, ref0 []byte, ref0Stride int, ref1 []byte, ref1Stride int) int {
	ctx := sad8x8CompoundAvgNEONCtx{
		Src:        unsafe.Pointer(&src[0]),
		Ref0:       unsafe.Pointer(&ref0[0]),
		Ref1:       unsafe.Pointer(&ref1[0]),
		SrcStride:  int64(srcStride),
		Ref0Stride: int64(ref0Stride),
		Ref1Stride: int64(ref1Stride),
	}
	sad8x8CompoundAvgNEONAsm(&ctx)
	return int(ctx.Sum)
}

func init() {
	sad8x8Impl = sad8x8NEON
	sad16x16Impl = sad16x16NEON
	sad32x32Impl = sad32x32NEON
	sad8x8x4Step4Impl = sad8x8x4Step4NEON
	sad8x8x4Impl = sad8x8x4NEON
	sad16x16x4Impl = sad16x16x4NEON
	sad16x16x4Step4Impl = sad16x16x4Step4NEON
	sad32x32x4Impl = sad32x32x4NEON
	sad32x32x4Step4Impl = sad32x32x4Step4NEON
	if useDotProdSAD {
		sad16x16x4Impl = sad16x16x4DotProd
		sad32x32x4Impl = sad32x32x4DotProd
		sad16x16x4Step4Impl = sad16x16x4Step4DotProd
		sad32x32x4Step4Impl = sad32x32x4Step4DotProd
	}
	sad8x8DualImpl = sad8x8DualNEON
	sad16x16DualImpl = sad16x16DualNEON
	sad32x32DualImpl = sad32x32DualNEON
	sad8x8CompoundAvgBlockImpl = sad8x8CompoundAvgBlockNEON
}
