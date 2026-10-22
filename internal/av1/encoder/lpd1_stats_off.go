//go:build !goav1_encstats

package encoder

const lpd1StatsEnabled = false

func noteLPD1SBLevel(level lpd1PD1Level) {}

func dumpLPD1Stats(frame uint64) {}
