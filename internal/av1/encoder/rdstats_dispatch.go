package encoder

// residualBlockImpl extracts the prediction residual (scalar reference unless
// the Go SIMD kernels bind it in rdstats_gosimd.go).
var residualBlockImpl = residualBlockPureGo

// rdStatsBlockImpl accumulates the skip-decision statistics (scalar reference
// unless the Go SIMD kernels bind it in rdstats_gosimd.go).
var rdStatsBlockImpl = rdStatsBlockPureGo

// blockErrorImpl computes the coefficient block error (scalar reference unless
// the Go SIMD kernels bind it in rdstats_gosimd.go).
var blockErrorImpl = blockErrorPureGo
