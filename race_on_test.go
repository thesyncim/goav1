//go:build race

package goav1

// raceEnabled reports whether this test binary was built with the race
// detector. Race instrumentation slows the codec several times over and
// allocates on its own schedule, so CPU-ratio gates skip under it.
const raceEnabled = true
