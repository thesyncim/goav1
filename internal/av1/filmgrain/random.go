// Ported from libaom: av1/decoder/grain_synthesis.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package filmgrain

const (
	CbSeedXor = 0xb524
	CrSeedXor = 0x49d8
)

// Random is the 16-bit AV1 film grain pseudo-random generator state.
type Random struct {
	register uint16
}

// NewRandom returns a film grain random generator seeded for picture-level use.
func NewRandom(seed uint16) Random {
	return Random{register: seed}
}

// NewPlaneRandom returns a film grain random generator seeded for a chroma
// grain plane. Plane 1 is Cb and plane 2 is Cr.
func NewPlaneRandom(seed uint16, plane int) (Random, error) {
	switch plane {
	case 1:
		return Random{register: seed ^ CbSeedXor}, nil
	case 2:
		return Random{register: seed ^ CrSeedXor}, nil
	default:
		return Random{}, ErrInvalidParams
	}
}

// NewStripeRandom returns a random generator seeded for the add-noise stripe
// that contains lumaLine.
func NewStripeRandom(seed uint16, lumaLine int) (Random, error) {
	if lumaLine < 0 {
		return Random{}, ErrInvalidParams
	}
	lumaNum := lumaLine >> 5
	register := seed
	register ^= uint16(((lumaNum*37 + 178) & 255) << 8)
	register ^= uint16((lumaNum*173 + 105) & 255)
	return Random{register: register}, nil
}

// Register reports the current 16-bit generator state.
func (r Random) Register() uint16 {
	return r.register
}

// Number advances the generator once and returns the requested high bits.
func (r *Random) Number(bits uint8) (uint16, error) {
	if bits == 0 || bits > 16 {
		return 0, ErrInvalidParams
	}
	bit := ((r.register >> 0) ^ (r.register >> 1) ^ (r.register >> 3) ^ (r.register >> 12)) & 1
	r.register = (r.register >> 1) | (bit << 15)
	if bits == 16 {
		return r.register, nil
	}
	return (r.register >> (16 - bits)) & ((1 << bits) - 1), nil
}
