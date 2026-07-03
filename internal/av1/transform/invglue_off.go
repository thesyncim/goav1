//go:build (!arm64 && !amd64) || purego

package transform

// clampRoundImpl rounds-and-shifts then clamps scratch in place.
var clampRoundImpl = clampRoundPureGo

// narrowStoreImpl writes the final residual rows.
var narrowStoreImpl = narrowStorePureGo
