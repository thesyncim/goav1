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
// Scan table (coeffScanHotTable[size][Class2D], 8 bytes per coefficient,
// loaded with a single MOVD per iteration and cracked with UBFX/SBFX):
//
//	bits  0-15 pos            raster position (col*height+row), <= 1023
//	bits 16-31 padded         levels-scratch offset col*stride+row
//	bits 32-39 lower2DOffset  signed base-context class offset {0,1,6,11,16,21}
//	bits 40-47 br2DOffset     signed BR-context offset {0,7,14}
//	bits 48-63 lowerEOBCtx/brEOBCtx (P0 only; ignored by the P1 kernel)
//
// Levels scratch (*uint8): padded column-major level buffer, stride =
// scanHeight+4; capacity (scanWidth+4)*stride proven by the Go prologue, so
// every neighbour read (padded+1, +2, +stride, +stride+1, +2*stride) is in
// bounds by construction — the asm carries no bounds checks, exactly like the
// `_ = levelsScratch[s2]` BCE hints it replaces.
//
// Outputs (append-only, capacity 1024 each, proven by eob <= maxEOB <= 1024):
//
//	dirtyArr      *[1024]int16  packed (level<<10 | pos), replayed by P3
//	levelDirtyArr *[1024]int16  padded offsets for the next-TXB sparse clear
//
// Geometry constants passed as scalars: cHi (= eob-2), cLo (0 for the fused
// loop+DC form), stride, update flag. Return value: number of appended
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
// 3. Register plan for the P1(+P2) kernel — coeffBaseLevels2DARM64
//
// func coeffBaseLevels2DARM64(c *Cursor, scanHot unsafe.Pointer, cHi, cLo int64,
//	levels *uint8, stride int64, base *CDF, br *CDF,
//	dirty *int16, levelDirty *int16, update int64) int64
//
// Loop-resident (live across iterations):
//
//	R1  baseArr        R2  brArr         R3  scanHot        R4  levels
//	R5  cnt (sx16)     R6  pos           R7  dif            R8  rng
//	R10 tellOffs(sx16) R19 c             R20 cLo            R21 stride
//	R22 dirty cursor   R23 levelDirty cursor                R24 update flag
//	R25 appended count R26 scanHot[c] raw / BR extra accumulator
//	R15 level accumulator for the current coefficient
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
// Not touched: V8-V15 (no vector code at all), R18 (platform), R27 (asm
// temp), R28 (g), R29/R30 (FP/LR). NOSPLIT, frame $0, args 96 bytes — leaf
// with zero locals, far inside the <800B nosplit headroom rule.
//
// Per-iteration flow:
//
//	load  R26 = MOVD (scanHot)(c<<3)
//	ctx:  pos==0 -> 0; else five MOVBU neighbour loads, clip3 via the
//	      branchless (x-3)&((x-3)>>63) identity folded to Σt+15, (mag+1)>>1,
//	      CSEL-min 4, + SBFX lower2DOffset
//	read: 4-symbol inverse-CDF search (three MUL/CMP steps, early-out BCS),
//	      renormalize (CLZ on rng low 16), refill iff cnt<0
//	adapt: skipped when update==0; else rate = 5 + count>>4, the symbol-indexed
//	      4-way branch updates c0..c2 in memory (up: v += (32768-v)>>rate,
//	      down: v -= v>>rate) and stores count+1 while count < 32 — the exact
//	      updateCDFWindow shape for symbols==4 (rate bias +1 applies because
//	      symbols > 3; 5 = 4+1 is precomputed)
//	BR:   if symbol==3: brCtx from three unclipped neighbour levels
//	      ((mag+1)>>1 CSEL-min 6 + SBFX br2DOffset), then the chained reads on
//	      row br+36*brCtx: extra += symbol, loop while symbol==3 && extra<12
//	      (extra==3k counts the reads, so no separate chain counter register),
//	      adapt-per-read iff update — bit-identical to both
//	      readCDF4HighTokenUpdateLoop and readCDF4HighTokenKnown because the
//	      no-update variant re-reads the unmodified row from memory
//	store: level==0 -> next; else MOVB level -> levels[padded], MOVH padded ->
//	      levelDirty++, MOVH (level<<10|pos) -> dirty++, appended++
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
//	readCoefficientsTXBWithGeo: the useScanHot cdfUpdate loop (c=eob-2..0).
//
// The Horiz/Vert and untrusted-scan loops stay in Go (different context
// derivation; D3-c decides whether they are worth their own kernels — they
// are a small share of coefficients).
//
// Dispatch is a package-level bool (static branch, no func pointers per the
// zero-alloc rules): entropy.HasCoeffBaseLevels2D (build-tag constant,
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
