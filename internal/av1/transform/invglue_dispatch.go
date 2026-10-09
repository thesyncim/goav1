package transform

// clampRoundImpl rounds-and-shifts then clamps scratch in place. Go SIMD
// builds rebind it in invglue_gosimd_{arm64,amd64}.go.
var clampRoundImpl = clampRoundPureGo

// narrowStoreImpl writes the final residual rows. Go SIMD builds rebind it in
// invglue_gosimd_{arm64,amd64}.go.
var narrowStoreImpl = narrowStorePureGo
