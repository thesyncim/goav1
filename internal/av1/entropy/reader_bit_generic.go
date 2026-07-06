// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build !arm64 || purego || goav1_trace_rng

package entropy

// ReadBitTrusted decodes one equiprobable bit from cursor state.
//
//go:nosplit
func (c *Cursor) ReadBitTrusted() uint8 {
	if traceEntropyReads {
		return c.readBitTrustedTrace()
	}
	dif := c.dif
	rangeValue := uint32(c.rng)
	cnt := int32(c.cnt)
	split := (rangeValue >> 8) << 7
	split += ecMinProb
	window := uint64(split) << (ecWindow - 16)

	bit := uint8(1)
	nextRange := split
	if dif >= window {
		dif -= window
		nextRange = rangeValue - split
		bit = 0
	}

	shift := normShift16(nextRange)
	cnt -= shift
	dif = ((dif + 1) << uint(shift)) - 1
	rangeValue = nextRange << uint(shift)
	if cnt < 0 {
		pos := int(c.pos)
		tellOffs := int32(c.tellOffs)
		src := c.src
		pos, dif, cnt, tellOffs = refillState(src, pos, dif, cnt, tellOffs)
		c.pos = uint32(pos)
		c.tellOffs = int16(tellOffs)
	}
	c.cnt = int16(cnt)
	c.dif = dif
	c.rng = uint16(rangeValue)
	return bit
}
