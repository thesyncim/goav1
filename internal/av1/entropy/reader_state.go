// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package entropy

// ReaderState is a snapshot of the range-decoder state used by differential
// tests (the M-D3 TXB asm harness compares pure-Go and kernel decodes
// bit-for-bit). It intentionally exposes every field that the decode math
// depends on; two readers with equal ReaderState over the same src decode
// identically forever.
type ReaderState struct {
	Pos      uint32
	Dif      uint64
	Rng      uint16
	Cnt      int16
	TellOffs int16
}

// State returns the range-decoder state snapshot for differential tests.
func (r *Reader) State() ReaderState {
	if r == nil {
		return ReaderState{}
	}
	return ReaderState{
		Pos:      r.pos,
		Dif:      r.dif,
		Rng:      r.rng,
		Cnt:      r.cnt,
		TellOffs: r.tellOffs,
	}
}
