---
name: optimize-go-simd
description: Make a Go-native SIMD kernel (GOEXPERIMENT=simd, simd/archsimd) match or beat its hand-written assembly. Use when a Go SIMD kernel is slower than the asm/C reference, or when writing a new one. Debug the disassembly and apply the patterns below — "codegen-bound / can't beat asm" is almost always one of these fixable causes.
---

# Optimizing Go-native SIMD kernels

A Go SIMD kernel being slower than its hand asm is almost never a fundamental
"register-allocation wall." Repeatedly, the real cause is one of the concrete,
fixable patterns below. Diagnose from the disassembly, fix, re-measure, repeat.
Do not accept a slower kernel and do not conclude "codegen-bound" until every
pattern here is ruled out.

## Always start from the disassembly

```
go tool objdump -s "kernelName" ./the.test | grep -oE '\t[A-Z][A-Z0-9_.]+' | sort | uniq -c | sort -rn
```

Tell-tales and what they usually mean:

| symptom in the hot function | real cause |
|---|---|
| many `VMOV` (reg↔reg moves) | a *symptom* of one of the causes below, not the cause |
| `VDUP` inside the loop | constants rebuilt every iteration (not hoisted) |
| `VSSHL`/`VUSHL` where the shift is constant | runtime shift rebuilding a shift-amount vector |
| lots of `STP`/`FMOVQ` to/from `RSP` | vectors bouncing through stack arrays between passes |
| `CALL runtime.panicBounds` | per-iteration slice bounds checks |
| `CALL` to your own helpers in the loop | helper too big to inline → args/returns via the stack ABI |

Also check the CALL targets specifically:
```
go tool objdump -s "kernelName" ./the.test | grep '\tCALL' | sed -E 's/.*CALL\t*//' | sort | uniq -c
```

## The patterns, roughly by impact

1. **Constant shifts must use the immediate shift, not the runtime shift.** A
   shift by a compile-time constant compiled through the general (runtime-amount)
   shift rebuilds a shift-amount vector (broadcast + move + negate + select) on
   *every use*. Use the immediate-shift form (`SSHR`/`SHL #imm`). This alone
   often removes most of the `VDUP`/`VMOV` churn.
   - Gotcha: an immediate shift only lowers correctly when the amount is a
     compile-time constant **at the lowering point**. If it flows through a
     non-inlined function as a parameter it can silently produce wrong results.
     Keep the constant as a literal in the function that emits the shift.

2. **Loop-invariant runtime shift → hoist the amount vector.** If the amount is
   a parameter that's constant for the whole call, build the amount vector once
   before the loop and use variable-shift-by-vector, instead of the runtime
   shift rebuilding the amount every iteration.

3. **Constants: build them as locals, never round-trip through memory.** A
   helper that *returns a struct of vectors* (or takes one by value) forces a
   stack round-trip — the constants get reloaded every iteration. Build them as
   plain local variables before the loop so they stay register-resident. If a
   struct is unavoidable, an inline struct literal lets SROA scalar-replace it;
   a value returned from a non-inlined function will not.

4. **Keep vectors register-resident across passes.** Passing `[N]Vec` arrays
   between passes (column → transpose → row) forces the whole set onto the stack
   (the ABI can't keep an array in registers). Chain passes with *individual*
   vector values, or inline the passes into one straight-line function, so the
   compiler keeps them in registers. This can be the single biggest win for
   multi-pass transforms.
   - No function calls in the vector body — inline every pass **textually**. A
     call clobbers all caller-saved V-registers, so the caller spills its entire
     live vector set around it; and a helper *returning* 8+ vectors (a butterfly
     or a two-chain filter) exceeds the inline budget, so it stays a real call
     even with individual (non-array) args. Verify with the CALL-target dump
     (`objdump | grep '\tCALL'`) that nothing in the hot body calls your own
     code; if the compiler won't inline it, inline it by hand. (This took the
     loop-filter filter4 and the forward DCT/ADST from parity to beating asm.)

5. **Load coefficient/twiddle tables from rodata.** A scalar broadcast is
   ~3 instructions (immediate-move + move + dup). A pre-broadcast package-level
   constant array loaded with one vector load is far cheaper. Prefer a
   `var tbl = [...]T{...}` loaded once over `Broadcast(scalar)` per use.

6. **Raw pointers in the hot loop.** Slice indexing bounds-checks every access.
   Walk a raw pointer (`unsafe.Pointer` + `unsafe.Add`) in the vector loop; keep
   checked indexing only in the scalar tail. (Measure — sometimes the checks are
   well-predicted and this is minor; the shift/constant fixes usually matter more.)

7. **Branchless over select.** Replace a zero-broadcast + compare + select with
   arithmetic where cheaper — e.g. conditional negate via a sign mask:
   `x ^ signMask - signMask` where `signMask = src >> (width-1)`.

8. **Multiple independent accumulators in reductions.** Use ≥2 accumulators so
   the abs→add / mul→add chains pipeline instead of forming one serial
   dependency chain. This frequently beats asm outright, no other change needed.

9. **Don't out-guard the asm.** If the reference asm scans nothing for a rare
   case (overflow, extreme inputs), don't add a per-iteration guard. Reproduce
   the reference's *exact* arithmetic — including deliberate overflow behavior —
   so you stay byte-exact without the guard.

## Byte-exact is the gate

Every change must be verified byte-identical to the scalar/asm reference with a
differential test, in both the `GOEXPERIMENT=simd` and plain builds. Harden the
differential test with extreme inputs (min/max of the type, values past the
"safe" range): a guard you remove may have been masking a real divergence.
Beating the asm at the wrong answer is not beating the asm.

## Also verify

- Zero heap escapes in the vector body (`-gcflags=-m`); array-pointer loads, not
  slices, inside the loop.
- The kernel is actually *dispatched* in the SIMD build (a benchmark that calls
  it directly does not prove the dispatch binds it) — probe the bound function
  with `runtime.FuncForPC`.
