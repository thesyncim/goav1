# itxgen

`go run ./internal/av1/transform/itxgen [adst|dct8]` prints the eight-lane
int16 butterfly networks of the fused inverse-transform kernels
(itx_fused*_gosimd_arm64.go) from the stage structure of the scalar
transforms in dct.go / adst.go, folding the zero operands of each
eob-bounded form symbolically and memoising repeated rotations:

- no argument: the DCT64 odd half (itx64Odd32/16/8, itx_fused64)
- `adst`: ADST16 forms 16/8/4 and ADST8 forms 8/4 (itx_fused_hybrid)
- `dct8`: the DCT8 Low4 form (itx_fused8_low4)

Constants come from itx64K (cospi index, negative at 64+i). A generated
network is certified by the float gain model of its transform
(itx*_gain_test.go: largest single-input gain x the per-lane L1 guard plus
the rounding slack must stay inside int16) and by the differential test of
its kernel against the staged path.
