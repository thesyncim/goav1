// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package decoder

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/av1/obu"
)

// fuzzMaxUnits bounds the number of OBU/IVF units pulled from a single fuzz
// input so a hostile stream can never make the harness loop forever. Every
// well-formed stream the conformance suite decodes stays far below this.
const fuzzMaxUnits = 1 << 16

// fuzzStreamSeeds are small, self-contained byte streams that exercise the
// untrusted-input parsing surface. They mirror the synthetic vectors used by
// the conformance core suite so the corpus is meaningful even in checkouts
// where the downloaded libaom .ivf vectors are absent.
var fuzzStreamSeeds = [][]byte{
	// IVF container wrapping a single low-overhead temporal-delimiter OBU.
	{
		'D', 'K', 'I', 'F',
		0x00, 0x00,
		0x20, 0x00,
		'A', 'V', '0', '1',
		0x10, 0x00,
		0x10, 0x00,
		0x1e, 0x00, 0x00, 0x00,
		0x01, 0x00, 0x00, 0x00,
		0x01, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x02, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x12, 0x00,
	},
	// Bare low-overhead temporal-delimiter OBU.
	{0x12, 0x00},
	// Full sequence header (low-overhead).
	{0x0a, 0x0b, 0x00, 0x00, 0x00, 0x04, 0x45, 0x7e, 0x3e, 0xff, 0xfc, 0xc0, 0x20},
	// Reduced still-picture sequence header (low-overhead).
	{0x0a, 0x07, 0x18, 0x22, 0x2b, 0xf1, 0xfe, 0xc0, 0x20},
	// Annex B temporal unit with two frame units.
	{0x0a, 0x05, 0x01, 0x10, 0x02, 0x08, 0xaa, 0x03, 0x02, 0x18, 0xbb},
	// Sequence header followed by a temporal delimiter (low-overhead).
	{0x0a, 0x0b, 0x00, 0x00, 0x00, 0x04, 0x45, 0x7e, 0x3e, 0xff, 0xfc, 0xc0, 0x20, 0x12, 0x00},
}

// addIVFSeedFiles best-effort seeds the corpus from any .ivf vectors present in
// the repository testdata tree. The libaom vectors are gitignored and only
// exist in checkouts that have downloaded them, so their absence is not fatal:
// the synthetic seeds above keep the corpus non-empty everywhere.
func addIVFSeedFiles(f *testing.F) {
	for _, root := range []string{
		filepath.Join("..", "testdata"),
		filepath.Join("..", "testdata", "libaom"),
	} {
		entries, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if e.IsDir() || filepath.Ext(e.Name()) != ".ivf" {
				continue
			}
			data, err := os.ReadFile(filepath.Join(root, e.Name()))
			if err != nil {
				continue
			}
			f.Add(data)
		}
	}
}

// FuzzStreamPush feeds arbitrary bytes through every untrusted-input entry
// point the decoder exposes for a byte stream: the IVF demuxer, the
// low-overhead / Annex B / temporal-unit OBU iterators, the single-OBU parser,
// and decoder.Stream's OBU dispatch (which transitively drives every
// parser.ParseXxx frame-header routine). It asserts the decoder never panics
// and always terminates on corrupt, truncated, or hostile input.
func FuzzStreamPush(f *testing.F) {
	for _, seed := range fuzzStreamSeeds {
		f.Add(seed)
	}
	addIVFSeedFiles(f)

	f.Fuzz(func(t *testing.T, data []byte) {
		// A panic anywhere in the parse path is a hardening defect: malformed
		// input must surface as an error, never a crash.
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panic on malformed input (len=%d): %v", len(data), r)
			}
		}()

		// Path 1: treat the input as an IVF container and push each demuxed
		// temporal-unit payload through the stream as low-overhead OBUs.
		fuzzDriveIVF(data)

		// Path 2..n: treat the whole input as a raw OBU byte stream in each of
		// the supported framings.
		fuzzDriveLowOverhead(data)
		fuzzDriveAnnexB(data)
		fuzzDriveTemporalUnit(data)
		fuzzDriveSingleElement(data)
	})
}

func fuzzDriveIVF(data []byte) {
	it, err := ivf.NewIterator(data)
	if err != nil {
		return
	}
	var stream Stream
	var events [256]Event
	frames := 0
	for {
		frame, ok, err := it.Next()
		if err != nil || !ok {
			return
		}
		frames++
		if frames > fuzzMaxUnits {
			return
		}
		if _, err := stream.PushLowOverhead(frame.Payload, events[:]); err != nil {
			// A bad payload aborts the stream; this mirrors how a real caller
			// would stop decoding on the first error.
			return
		}
	}
}

func fuzzDriveLowOverhead(data []byte) {
	it := obu.NewLowOverheadIterator(data)
	var stream Stream
	var event Event
	units := 0
	for {
		unit, ok, err := it.Next()
		if err != nil || !ok {
			return
		}
		units++
		if units > fuzzMaxUnits {
			return
		}
		if err := stream.PushUnitInto(&event, unit, units == 1); err != nil {
			return
		}
	}
}

func fuzzDriveAnnexB(data []byte) {
	it := obu.NewAnnexBIterator(data)
	var stream Stream
	var event Event
	units := 0
	for {
		unit, ok, err := it.Next()
		if err != nil || !ok {
			return
		}
		units++
		if units > fuzzMaxUnits {
			return
		}
		if err := stream.PushUnitInto(&event, unit.OBU, false); err != nil {
			return
		}
	}
}

func fuzzDriveTemporalUnit(data []byte) {
	it := obu.NewTemporalUnitIterator(data)
	tus := 0
	for {
		tu, ok, err := it.Next()
		if err != nil || !ok {
			return
		}
		tus++
		if tus > fuzzMaxUnits {
			return
		}
		fuzzDriveLowOverhead(tu.Raw)
	}
}

func fuzzDriveSingleElement(data []byte) {
	unit, err := obu.ParseElement(data)
	if err != nil {
		return
	}
	var stream Stream
	var event Event
	_ = stream.PushUnitInto(&event, unit, true)
}
