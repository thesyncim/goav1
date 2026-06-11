//go:build !arm64 || purego

package encoder

// residualBlockImpl extracts the prediction residual (portable form).
var residualBlockImpl = residualBlockPureGo

// rdStatsBlockImpl accumulates the skip-decision statistics (portable form).
var rdStatsBlockImpl = rdStatsBlockPureGo
