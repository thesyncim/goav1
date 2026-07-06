//go:build race

package encoder_test

// raceDetectorEnabled reports whether this test binary was built with the race
// detector. The race runtime allocates on its own schedule (shadow bookkeeping,
// channel-park sudogs survive differently), so strict TotalAlloc==0 guards are
// unreliable under it.
const raceDetectorEnabled = true
