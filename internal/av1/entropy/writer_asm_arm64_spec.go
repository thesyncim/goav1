package entropy

import "unsafe"

// M-B0 writer-spine ARM64 specification.
//
// This file is the enabling spec for the encoder entropy Writer asm program.
// It is intentionally source-shaped against the existing Go writer and the
// pinned SVT-AV1 v4.1.0 range encoder:
//
//   - third_party/upstream/svt-av1/Source/Lib/Codec/bitstream_unit.c:
//     svt_od_ec_enc_normalize, svt_od_ec_encode_q15,
//     svt_od_ec_encode_bool_q15, svt_od_ec_encode_cdf_q15.
//   - local writer.go: normalize, encodeQ15, WriteBoolQ15, WriteCDF4.
//   - local tile/coeff_write.go: the per-TXB coefficient write body after the
//     eob token, especially the base/BR/sign/golomb loops.
//
// The decoder D3 chapter proved the useful cut: keep the arithmetic coder
// state resident across a whole TXB run, not across one symbol. This writer
// program mirrors that shape for the serial encoder tail.
//
// 0. Boundary decision
//
// Rejected: per-symbol Writer asm leaves the Go ABI call and Writer memory
// round-trip around every WriteCDF4/WriteBit. That is the tax this slice exists
// to remove, and the decoder D3 measurements already pinned primitive-only asm
// as too small by itself.
//
// Deferred: a whole coefficient writer including txb_skip, tx_type callback,
// eob-token selection and geometry validation. That would mix syntax decisions,
// error handling, callback ordering and range-coder mechanics into one first
// kernel.
//
// Chosen for B2: one kernel per TXB over the already-prepared coefficient
// write spine:
//
//   - base-eob and base symbol writes,
//   - the BR high-token chain,
//   - sign writes, including DCSign CDF adaptation,
//   - Exp-Golomb coefficient tails.
//
// Go keeps txb_skip/eob token/tx_type ordering, coefficient prep, scan table
// selection and validation. The kernel consumes the same scan-hot, levels,
// abs-level and non-zero/sign-bit images used by the 8x8/16x16/32x32 trusted
// writers. Count-only BitCounter paths stay in Go unless a separate counter
// kernel is justified by profile data.
//
// 1. Writer memory contract
//
// Writer is the only mutable range-coder state the writer kernel owns. The
// offsets below are hardcoded by the future arm64 text and are pinned by the
// compile-time assertions after this comment, using the same pattern as
// reader_coeff_base_arm64.go.
//
//   +0  buf.data (8)   output backing array pointer
//   +8  buf.len  (8)   writable length after any Go-side reserve
//   +16 buf.cap  (8)   not read by the leaf kernel
//   +24 offs     (8)   next byte to write
//   +32 low      (8)   od_ec_enc low window
//   +40 rng      (4)   range, maintained in [0x8000, 0xffff]
//   +44 cnt      (4)   buffered-bit count, initialized to -9
//   +48 err      (16)  sticky Go error interface
//
// The asm fast path is entered only when err == nil and buf is non-nil. The Go
// wrapper must reserve enough output bytes before entering so the NOSPLIT leaf
// never allocates. If the reserve cannot be proven for a future general kernel,
// dispatch falls back to the pure-Go TXB writer for that block before any CDF
// mutation occurs.
//
// CDF rows use the existing entropy.CDF layout also consumed by the decoder
// kernels: symbols at +0, values[0] at +2, row size 36 bytes. The writer kernel
// updates rows in place with the same updateCDFWindow arithmetic as WriteCDF4,
// WriteCDF3 and WriteBinaryCDFTrusted.
//
// 2. Shared normalize/writeOut/propagateCarry sub-block
//
// The writer-side analog of the decoder refill block is a textual asm
// sub-block shared by B1 and B2. It must stay instruction-for-instruction
// equivalent wherever copied:
//
//   d = CLZ16(rng)
//   s = cnt + d
//   if s >= 40:
//       numBytesReady = (s >> 3) + 1
//       c = cnt + 24 - (numBytesReady << 3)
//       keepMask = (1 << c) - 1
//       output = low >> c
//       low &= keepMask
//       carryMask = 1 << (numBytesReady << 3)
//       carry = output & carryMask
//       output &= carryMask - 1
//       store output as big-endian bytes at buf+offs, using the same 8-byte
//       store shape as writeOut
//       if carry != 0: propagateCarry from offs-1 toward zero
//       offs += numBytesReady
//       s = c + d - 24
//   low <<= d
//   rng <<= d
//   cnt = s
//
// propagateCarry is the only loop in the sub-block: increment buf[i], walk
// backward while the increment overflows, and set err = ErrCarryUnderflow if it
// would cross before byte zero. Hitting that error is a correctness failure for
// a well-formed stream, but the asm must preserve the Go behavior exactly.
//
// 3. Register plan for the TXB writer kernel
//
// Proposed ABI0 shape, refined when B2 lands:
//
//   func writerCoeffTXBARM64(
//       w *Writer,
//       scanHot unsafe.Pointer,
//       eob int,
//       levels *uint8,
//       absLevels *uint16,
//       nonZeroBits *uint64,
//       signBits *uint64,
//       stride int,
//       baseEOB *CDF,
//       base *CDF,
//       br *CDF,
//       dcSign *CDF,
//   )
//
// Loop-resident registers:
//
//   R1  Writer*       R2  scanHot       R3  levels        R4  absLevels
//   R5  low           R6  rng           R7  cnt           R8  offs
//   R9  buf.data      R10 buf.len       R19 c             R20 eob
//   R21 stride        R22 baseEOB       R23 base          R24 br
//   R25 dcSign        R26 current scan-hot word / level
//
// Scratch registers R0, R11-R17 are available to inverse-CDF search, row
// addressing, CDF adaptation and the normalize sub-block. R18 is left alone for
// platform use; R27/R28/R29/R30 keep their usual assembler/runtime roles.
//
// Base pass, reverse scan order:
//
//   - c == eob-1 uses baseEOB[lowerEOBCtx] and writes min(level, 3)-1 through
//     the 3-symbol CDF update shape.
//   - other coefficients use base[ctx] and write min(level, 3) through the
//     4-symbol update shape; known zero uses the WriteCDF4Zero arithmetic.
//   - ctx comes from the scan-hot offsets and levels neighbours exactly as in
//     coeff_write.go. For specializations that precompute lowerContexts, the Go
//     wrapper may pass that image; otherwise the kernel derives the 2D context
//     from levels.
//   - if level > NumBaseLevels, the BR loop repeatedly writes k in [0,3] to one
//     4-symbol CDF row until k < 3 or the base-range cap is reached.
//
// Sign/golomb pass, forward scan order:
//
//   - nonZeroBits selects the coefficients to replay.
//   - the DC coefficient writes through dcSign with binary-CDF adaptation.
//   - all other coefficients write one equiprobable sign bit.
//   - levels >= MaxBaseBRRange append the AV1 Exp-Golomb tail with the same bit
//     order as tile.writeGolomb.
//
// 4. Dispatch and kill switch
//
// B1/B2 should wire static bool dispatch, not function pointers:
//
//   writerAsmKillSwitch = os.Getenv("GOAV1_DISABLE_WRITER_ASM")
//   writerNormalizeKernel = HasWriterNormalize && writerAsmKillSwitch == ""
//   writerCoeffTXBKernel  = HasWriterCoeffTXB  && writerAsmKillSwitch == ""
//
// A non-empty GOAV1_DISABLE_WRITER_ASM value forces the pure-Go writer spine in
// the same binary. If an incremental landing needs sub-kernel measurement, named
// values such as "norm" or "txb" may temporarily disable only that stage, but
// the final soak should remove named measurement hooks and keep one documented
// kill switch while the asm chapter is active.
//
// 5. Differential harness
//
// writer_asm_diff_test.go is the B0 apparatus. Until a kernel exists, both
// switch positions intentionally route to the pure-Go writer. The lockstep
// record compares:
//
//   - Writer state before Finish, including buf/offs/low/rng/cnt/err,
//   - Tell after the write stream,
//   - Finish bytes and Writer state after Finish,
//   - the complete post-write CDF image.
//
// Every generated stream also round-trips through Reader with the same initial
// CDF image. GOAV1_WRITER_DIFF scales the number of randomized streams for B1
// and B2 soak runs.

var (
	_ [unsafe.Offsetof(Writer{}.buf)]byte
	_ [unsafe.Offsetof(Writer{}.offs) - 24]byte
	_ [24 - unsafe.Offsetof(Writer{}.offs)]byte
	_ [unsafe.Offsetof(Writer{}.low) - 32]byte
	_ [32 - unsafe.Offsetof(Writer{}.low)]byte
	_ [unsafe.Offsetof(Writer{}.rng) - 40]byte
	_ [40 - unsafe.Offsetof(Writer{}.rng)]byte
	_ [unsafe.Offsetof(Writer{}.cnt) - 44]byte
	_ [44 - unsafe.Offsetof(Writer{}.cnt)]byte
	_ [unsafe.Offsetof(Writer{}.err) - 48]byte
	_ [48 - unsafe.Offsetof(Writer{}.err)]byte
)
