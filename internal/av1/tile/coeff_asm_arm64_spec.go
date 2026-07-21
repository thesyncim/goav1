package tile

// M-D3 TXB-kernel specification (D3-a).
//
// This file is documentation only (no code). It is the register/memory plan
// for lowering the readCoefficientsTXB spine to scalar arm64 assembly, one
// conformance-safe kernel at a time. The companion differential harness lives
// in coeff_asm_diff_test.go. dav1d reference: src/arm/64/msac.S
// (msac_decode_symbol_adapt4 / msac_decode_hi_tok) and src/decode.c
// decode_coefs; goav1 already carries the 64-bit-window scalar kernels
// readBitTrustedARM64 and readCDF4HighTokenUpdateArch in
// internal/av1/entropy/reader_bit_arm64.s, whose refill block and
// operand-order conventions this spec reuses verbatim.
//
// ─────────────────────────────────────────────────────────────────────────────
// 0. What the spine is
//
// readCoefficientsTXBWithGeo / readCoefficientsTXBTracked2DWithGeo decode one
// transform block in four phases:
//
//	P0 eob-class entry  readEOBCursorKnown: one 5..11-symbol CDF read
//	                    (EOBFlag16..1024), one optional binary EOBExtra read,
//	                    then offsetBits-1 equiprobable bits. Then the
//	                    eob-position coefficient: one 3-symbol CoeffBaseEOB
//	                    read (+ BR chain if it saturates).
//	P1 base-levels loop for c = eob-2 .. 0 (tracked splits DC out, but the
//	                    fused body is identical): derive the base context from
//	                    five levels-scratch neighbours, one 4-symbol CoeffBase
//	                    read (+ adapt), and if the symbol saturates (==3) the
//	                    BR chain (P2) inline; write the level to the padded
//	                    levels scratch and append (pos, level) to the dirty
//	                    scan list and padded offset to the level-dirty list.
//	P2 BR loop          readCDF4HighToken*: up to four chained 4-symbol
//	                    CoeffBR reads on one row (level += symbol, stop when
//	                    symbol < 3), adapting the row between reads when CDF
//	                    update is on. Already a scalar arm64 kernel for the
//	                    update variant (readCDF4HighTokenUpdateArch).
//	P3 golomb/DC-sign   replay the dirty scan list from the DC end: one binary
//	                    DCSign read for pos 0, one equiprobable bit per other
//	                    coefficient, plus the Exp-Golomb tail
//	                    (ReadUnsignedGolombTrusted, maxLength 20) whenever the
//	                    base+BR level saturates at MaxBaseBRRange (15); apply
//	                    signs, accumulate culLevel, store int16 coeffs.
//
// Cost split (p720_inter_q32 profile): P1 dominates (it executes once per
// coefficient before eob and each iteration re-enters the Cursor read helpers,
// spilling dif/rng/cnt through memory per symbol); P2 is second but its body
// is already asm — the remaining cost is its per-chain call boundary; P0 and
// P3 execute once per TXB / once per nonzero coefficient respectively.
//
// ─────────────────────────────────────────────────────────────────────────────
// 1. Inputs and their memory shapes (all offsets pinned by compile-time
//    assertions in internal/av1/entropy/reader_coeff_base_arm64.go)
//
// Cursor (entropy.Cursor, 48 bytes; the tile spine snapshots s.Reader into a
// stack Cursor for the whole TXB and commits back at exit):
//
//	+0  src ptr   (8)  tile payload base
//	+8  src len   (8)
//	+16 src cap   (8)  unused by asm
//	+24 dif       (8)  64-bit coding window (libaom XOR-into-ones convention)
//	+32 pos       (4)  next buffered byte
//	+36 rng       (2)  range, 1..0xffff
//	+38 cnt       (2)  buffered-bit count, refill when < 0
//	+40 tellOffs  (2)  ecLotsBits tell accounting for past-end reads
//	+42 allowCDFUpdate (1) — NOT read by the kernel; the caller resolves the
//	    update mode once per TXB and passes it as an argument, matching the
//	    cdfUpdate hoisting already done in Go.
//
// CDF rows (entropy.CDF, 36 bytes): symbols u8 at +0, values [17]u16 at +2.
// For the 4-symbol coefficient rows: c0/c1/c2 at +2/+4/+6, adaptation count
// at +10. Row arrays used by the kernel:
//
//	baseArr *[42]CDF  = &cdfs.CoeffBase[txCtx][plane]   row = base + 36*ctx
//	brArr   *[21]CDF  = &cdfs.CoeffBR[txBRCtx][plane]   row = br   + 36*brCtx
//
// Row addressing is two ALU ops: r9 = ctx + ctx<<3 (9*ctx); row = arr + r9<<2.
//
// Scan table (coeffScanHotTable[size][class], 8 bytes per coefficient,
// loaded with a single MOVD per iteration and cracked with UBFX/SBFX):
//
//	bits  0-15 pos            raster position (col*height+row), <= 1023
//	bits 16-31 padded         levels-scratch offset col*stride+row
//	bits 32-39 lowerOffset    base-context class offset (2D or 1D)
//	bits 40-47 brOffset       BR-context class offset {0,7,14}
//	bits 48-63 lowerEOBCtx/brEOBCtx (P0 only; ignored by the P1 kernel)
//
// Levels scratch (*uint8): padded column-major level buffer, stride =
// scanHeight+4; capacity (scanWidth+4)*stride proven by the Go prologue, so
// every 2D or 1D neighbour read (through padded+4 or padded+4*stride) is in
// bounds by construction — the asm carries no bounds checks, exactly like the
// Go BCE hints it replaces. While this kernel owns the window, each nonzero
// byte packs the raw 0..15 level in bits 0..3 and min(level,3) in bits 4..5.
// P1 consumes the cached high nibble, P2 masks the raw low nibble, and the
// next-TXB sparse clear treats the byte as opaque.
//
// Outputs (append-only, capacity 1024 each, proven by eob <= maxEOB <= 1024):
//
//	dirtyArr      *[1024]int16  packed (level<<10 | pos), replayed by P3
//	levelDirtyArr *[1024]int16  padded offsets for the next-TXB sparse clear
//
// Geometry constants passed as scalars: cHi (= eob-2), cLo (0 for the fused
// loop+DC form), stride, class, update flag. Return value: number of appended
// (dirty, levelDirty) pairs — the two lists always grow in lockstep.
//
// ─────────────────────────────────────────────────────────────────────────────
// 2. Go -> asm boundary options considered
//
// (a) per-symbol leaf kernels (dav1d's own cut: msac.S decode_symbol_adapt4
//     called per coefficient from C). Rejected as the *primary* cut for Go:
//     Go ABI0 assembly calls pass args/results through the stack and the
//     Cursor state through memory, so each symbol still pays the
//     call + state-roundtrip tax that is the very residue M-D3 attacks.
//     dav1d tolerates the boundary because a C call keeps MsacContext hot and
//     args in registers; Go does not. Measured precedent in-tree: the D1-a
//     leaf kernels moved per-kernel time -1.4..-3.3% but were e2e-flat.
//
// (b) whole-TXB kernel (P0+P1+P2+P3 in one TEXT). Maximal win but rejected
//     for the first landing: it would fold the eob-token switch, the class
//     dispatch, golomb validation-error unwinding, and the culLevel/result
//     packaging into one >1000-instruction asm body — too much surface to
//     bring up in one conformance gate, and P0/P3 are not where the time is.
//
// (c) RECOMMENDED (this program's D3-b): per-loop kernel for P1 with P2
//     inlined — one asm call per TXB covering the entire base-levels walk
//     (including the DC iteration, which is just the c==0 case of the same
//     body) with the BR chain lowered inline so a saturating symbol never
//     crosses back into Go. dif/rng/cnt/pos stay in registers for the whole
//     loop; CDF rows adapt in place in memory; the only boundary cost is one
//     stack call per TXB (amortized over eob coefficients). P0 and P3 stay in
//     Go and reuse the existing leaf kernels (readBitTrustedARM64,
//     readCDF4HighTokenUpdateArch) where they already apply. D3-c then lowers
//     P3 (sign bits + golomb replay) the same way, and D3-d decides whether
//     fusing P0+P1+P3 into a whole-TXB kernel still pays.
//
// ─────────────────────────────────────────────────────────────────────────────
// 3. Register plan for the P1(+P2) kernel — coeffBaseLevelsARM64
//
// func coeffBaseLevelsARM64(c *Cursor, scanHot unsafe.Pointer, cHi, cLo int64,
//	levels *uint8, stride int64, base *CDF, br *CDF,
//	dirty *int16, levelDirty *int16, class int64, update int64) int64
//
// Loop-resident (live across iterations):
//
//	R1  baseArr        R2  brArr         R3  scanHot        R4  levels
//	R5  cnt (sx16)     R6  pos           R7  dif            R8  rng
//	R10 tellOffs(sx16) R19 c             R20 cLo            R21 stride
//	R22 dirty cursor   R23 levelDirty cursor                R24 update flag
//	R25 appended count R26 scanHot[c] raw / BR extra accumulator
//	R27 transform class; R15 level accumulator for the current coefficient
//
// Iteration-scratch: R0, R9, R11-R14, R16, R17.
//	R14 holds the current CDF row pointer and survives refill; R16 holds the
//	decoded symbol and survives refill + adapt; R13 is `upper` during the
//	inverse-CDF search.
//
// Refill block (verbatim 64-bit-window port shared with reader_bit_arm64.s,
// two textual copies: after the base read and inside the BR chain) clobbers
// only {R0, R9, R11, R12, R13, R17}; the cursor pointer and src ptr/len are
// reloaded from the stack args inside the block (refill is rare), freeing
// three otherwise-persistent registers for the loop.
//
// Not touched: V8-V15 (no vector code at all), R18 (platform), R28 (g),
// R29/R30 (FP/LR). NOSPLIT, frame $0, args 104 bytes — leaf
// with zero locals, far inside the <800B nosplit headroom rule.
//
// Per-iteration flow:
//
//	load  R26 = MOVD (scanHot)(c<<3)
//	ctx:  load the five cached min(level,3) high nibbles (2D folds its two
//	      adjacent pairs into MOVHU loads; vertical folds p1..p4 into one
//	      MOVWU; horizontal keeps its strided MOVBU loads), sum,
//	      (mag+1)>>1, CSEL-min 4, + SBFX lowerOffset
//	read: 4-symbol inverse-CDF search (three MUL/CMP steps, early-out BCS),
//	      renormalize (rng is already zero-extended in 1..65535, so direct
//	      64-bit CLZ minus 48 is CLZ16 without a masking dependency), refill
//	      iff cnt<0
//	adapt: skipped when update==0; else rate = 5 + count>>4, the symbol-indexed
//	      4-way branch updates c0..c2 in memory (up: v += (32768-v)>>rate,
//	      down: v -= v>>rate) and stores count+1 while count < 32 — the exact
//	      updateCDFWindow shape for symbols==4 (rate bias +1 applies because
//	      symbols > 3; 5 = 4+1 is precomputed)
//	BR:   if symbol==3: brCtx from three low-nibble raw neighbour levels
//	      ((mag+1)>>1 CSEL-min 6 + SBFX brOffset), then the chained reads on
//	      row br+36*brCtx: extra += symbol, loop while symbol==3 && extra<12
//	      (extra==3k counts the reads, so no separate chain counter register),
//	      adapt-per-read iff update — bit-identical to both
//	      readCDF4HighTokenUpdateLoop and readCDF4HighTokenKnown because the
//	      no-update variant re-reads the unmodified row from memory
//	store: level==0 -> next; else pack level|min(level,3)<<4 -> levels[padded],
//	      MOVH padded -> levelDirty++, MOVH (level<<10|pos) -> dirty++,
//	      appended++
//	next: c--; loop while c >= cLo
//
// ─────────────────────────────────────────────────────────────────────────────
// 4. Where the kernel is wired (and the kill switch)
//
// The kernel replaces, byte-for-byte:
//
//	readCoefficientsTXBTracked2DWithGeo: both cdfUpdate variants of the
//	  c=eob-2..1 loop plus the separate DC block (kernel cLo=0 covers DC:
//	  scanHot[0].pos==0 -> ctx 0, br2DOffset 0, identical order of appends).
//	readCoefficientsTXBWithGeo: trusted scan-hot loops in either update mode
//	  (c=eob-2..0).
//
// The same trusted scan-hot cut covers Class2D, ClassHoriz, and ClassVert;
// their packed rows carry the appropriate neighbour-context offsets. Only
// untrusted-scan and public no-scratch fallbacks stay in Go.
//
// Dispatch is a package-level bool (static branch, no func pointers per the
// zero-alloc rules): entropy.HasCoeffBaseLevels (build-tag constant,
// arm64 && !purego && !goav1_trace_rng) && GOAV1_DISABLE_COEFF_ASM unset.
// The env var is the kill switch; the differential harness proves the two
// sides of the switch byte-identical.
//
// ─────────────────────────────────────────────────────────────────────────────
// 5. Differential harness (coeff_asm_diff_test.go) — commit 1 deliverable
//
// Extends the entropy-level differential pattern (reader_arch_diff_test.go:
// per-read state equality over random streams + adversarial CDF states) to
// whole-TXB records:
//
//	record = TXBDecodeResult + coeffs + dirty/levelDirty contents + full
//	         reader state (entropy.Reader.State() snapshot: pos/dif/rng/cnt/
//	         tellOffs) + the complete post-TXB CoeffCDFs image (adaptation
//	         included) — any symbol-sequence divergence is visible in at
//	         least one of these.
//
// Cases: all 19 transform sizes x {2D, Horiz, Vert} x {update, no-update} x
// stream shapes {random, 0x00, 0xFF, 0xAA, random-short tails 0..17 bytes
// hitting the ecLotsBits past-end accounting} x CDF states {defaults for all
// four q contexts, Init-seeded extreme rows (near-saturated / near-zero)
// warmed to randomized adaptation counts via CDF.Update}. Streams are decoded
// as back-to-back TXBs sharing one reader and one CDF set, so adaptation
// state chains across TXBs like a real tile.
//
// Cross-checks (all must produce identical records):
//
//	tracked2D vs WithGeo scanHot vs WithGeo untrusted-scan (three independent
//	Go bodies today; kernel-on vs kernel-off joins the set in D3-b),
//	and the golomb/BR/DC-sign paths are exercised by the near-saturated CDF
//	seeds (levels >= 15 force golomb).
//
// The default -count=1 run stays a few seconds; GOAV1_TXB_DIFF_TXBS scales
// the randomized-TXB volume to the millions for the D3-b landing gate.
//
// ─────────────────────────────────────────────────────────────────────────────
// 6. D3-c: the P3 sign/golomb replay kernel — coeffSignGolombARM64
//
// func coeffSignGolombARM64(c *Cursor, dirty *int16, n int, coeffs *int16,
//	dcSign *CDF, update bool) (culLevel, dcValue, maxScanLine int)
//
// One NOSPLIT leaf call per TXB replaying the whole dirty-scan list from
// index n-1 down to 0: the DC entry (pos==0; only ever the last-appended,
// because every scan places position 0 at scan index 0) reads its sign from
// the 2-symbol DC-sign row (readBinaryCDFKnown + updateCDF2 when update is
// set), every other entry reads one equiprobable bit (ReadBitTrustedInline
// shape), and a level saturated at MaxBaseBRRange appends the Exp-Golomb
// tail (unary prefix capped at maxLength 20, then length-1 suffix bits).
// Signed coefficients are stored, entries are unpacked to bare positions,
// and culLevel/dcValue/maxScanLine come back as scalars. culLevel == -1
// encodes the Go loop's two validation errors (prefix beyond maxLength,
// level > math.MaxInt16) with the cursor already at the failure point and
// the failing entry's stores skipped, so the Go wrapper's CommitStateTo +
// ErrInvalidDecodeState return is byte-identical to the pure loop.
//
// KEPT SEPARATE from the D3-b kernel (not fused) for this landing: §2's
// roadmap defers the whole-TXB fuse decision to D3-d, and the standalone P3
// cut serves strictly more call sites than a fuse could — the replay body is
// identical across Class2D/Horiz/Vert and both update modes, so the kernel
// wires into BOTH readCoefficientsTXBWithGeo (any class, any update mode,
// whenever the dirty-scan list is in use) and
// readCoefficientsTXBTracked2DWithGeo, including the Horiz/Vert and
// no-update paths that never run the base-levels kernel. The call boundary
// stays one stack call per TXB, amortized over the nonzero coefficients.
//
// Register plan mirrors §3: persistent R1 dirty, R2 coeffs, R4 culLevel,
// R5 cnt, R6 pos, R7 dif, R8 rng, R10 tellOffs, R19 i, R20 maxScanLine,
// R21 dcValue, R22 update, R23 level, R24 negative, R25 entry pos, R26 isDC;
// R14/R15 carry the golomb length/accumulator across bit reads (the refill
// clobber set {R0, R9, R11, R12, R13, R17} is unchanged, four textual
// copies). The suffix accumulates as acc = acc<<1|bit from acc = 1, which
// finishes as (1<<(length-1))|suffix so value = acc-1 without a separate
// shift register. Dispatch: coeffSignGolombKernel in coeff_asm_dispatch.go,
// same GOAV1_DISABLE_COEFF_ASM kill switch as D3-b ("1" kills both kernels;
// the value "sign" kills only this one — the incremental-measurement hook
// for its landing review, removed with the switch when M-D3 concludes).
//
// ─────────────────────────────────────────────────────────────────────────────
// 7. D3-e: the MV residual chain — mvResidualARM64
//
// func mvResidualARM64(c *Cursor, mvCDFs unsafe.Pointer, precision int64,
//	update bool) (joint, comp0, comp1 int64)
//
// The D3-c decision gate moved the spine program to the mode/MV symbol
// subtrees: with both TXB kernels active, the profile puts
// decodeBlockPredictionModeInto at ~10-11% cum on p720_inter_q32 /
// p288_inter_q20 — but its weight is glue (BuildReferenceMVStack ~3-4%,
// MarkInter* grid marking ~1-2%), not symbol reads. The contiguous
// symbol-read runs rank: ReadInterReferences (~0.5-1.0% cum, but every bit's
// context re-derives from BlockModeContext neighbour counts between reads —
// not self-contained), the MV residual chain (~0.3-0.6% cum, fully
// self-contained: CDF rows + cursor + small outputs), and the inter-mode
// cascade + DRL bits (~0.3%, contexts precomputed from the stack). The MV
// chain is the hottest kernelable run and the longest dependent sequence of
// small reads (joint 4-symbol + per component: sign binary, 11-symbol class,
// class-0 binary or up to ten binary integer bits, 4-symbol fraction, binary
// high-precision — up to 28 reads per compound NEWMV block), so it goes
// first; one NOSPLIT leaf call per ReadMotionVector replaces up to 15
// per-symbol Cursor helper calls with their dif/rng/cnt memory round-trips.
//
// The kernel (entropy/reader_mv_arm64.s) hardcodes the tile.MVCDFs offsets
// (Joint +0, Components[0/1] +36/+684; within a component Classes +0,
// Class0FP +36/+72, FP +108, Sign +144, Class0HP +180, HP +216, Class0 +252,
// Bits[i] +288+36i — pinned at compile time in mv_asm.go) and walks the two
// components as a loop selected by the joint bits. Register plan follows §3:
// persistent R1 mvCDFs, R5-R10 cursor window, R19 component index, R20
// component base, R21 precision, R22 update, R23/R24 packed results, R25
// joint; body locals R2 sign, R3 class, R4 fraction, R15 bits-index-then-mag,
// R26 integerPart; R14 row / R16 symbol survive the refills (same clobber
// set, seven textual copies). Component results return packed (diff int16 |
// integerPart<<16 | class<<26 | fraction<<30 | hp<<32 | sign<<33); the
// chain cannot fail (|diff| <= 16384 by construction, all row shapes
// pre-validated by the Go wrapper via mvCDFsKernelShape, which falls back to
// the pure chain — and its exact mid-chain error unwind — on any malformed
// row). Wired into DecodeState.ReadMotionVector for both call sites (NEWMV
// inter motion and intrabc DVs, which pass precision -1); dispatch:
// mvResidualKernel in coeff_asm_dispatch.go, same GOAV1_DISABLE_COEFF_ASM
// switch (any value kills all spine kernels; the value "mv" kills only this
// one — the incremental-measurement hook for its landing review).
//
// Differential: mv_asm_diff_test.go runs the kernel and the pure chain in
// lockstep (same stream, cloned reader + MVCDFs) over the txbDiffPayload
// stream shapes, three CDF seeds (defaults, scrambled-with-warmed-counts,
// tail-heavy joint+class rows that force both components, class 10, the full
// ten-bit integer loop and the ±16384 magnitude boundary), precisions
// -1/0/1, update on/off, and extreme refs that steer ref+diff into the
// clamp error path; compares mv, MVResidualResult, error, the reader state
// snapshot and the post-read MVCDFs image per read. Same GOAV1_TXB_DIFF_TXBS
// soak knob as the TXB harnesses.
