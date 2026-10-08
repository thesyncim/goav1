// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64

package cpu

import "testing"

// auxvImage builds a little-endian arm64 auxv byte image from tag/value pairs,
// AT_NULL-terminated, matching the /proc/self/auxv layout.
func auxvImage(pairs [][2]uint64) []byte {
	out := make([]byte, 0, (len(pairs)+1)*16)
	put := func(v uint64) {
		for i := 0; i < 8; i++ {
			out = append(out, byte(v>>(8*i)))
		}
	}
	for _, p := range pairs {
		put(p[0])
		put(p[1])
	}
	put(atNull)
	put(0)
	return out
}

func TestParseAuxvHWCap(t *testing.T) {
	hwcap, hwcap2 := parseAuxvHWCap(auxvImage([][2]uint64{
		{6, 4096}, // AT_PAGESZ — ignored
		{atHWCap, hwcapASIMDDP | hwcapSVE},
		{atHWCap2, hwcap2I8MM | hwcap2SVE2},
	}))
	if hwcap != hwcapASIMDDP|hwcapSVE {
		t.Fatalf("hwcap = %#x, want %#x", hwcap, hwcapASIMDDP|hwcapSVE)
	}
	if hwcap2 != hwcap2I8MM|hwcap2SVE2 {
		t.Fatalf("hwcap2 = %#x, want %#x", hwcap2, hwcap2I8MM|hwcap2SVE2)
	}
}

func TestParseAuxvHWCapStopsAtNull(t *testing.T) {
	// An entry positioned after AT_NULL must not be read.
	img := auxvImage([][2]uint64{{atHWCap, hwcapASIMDDP}})
	img = append(img, auxvImage([][2]uint64{{atHWCap2, hwcap2I8MM}})...)
	hwcap, hwcap2 := parseAuxvHWCap(img)
	if hwcap != hwcapASIMDDP {
		t.Fatalf("hwcap = %#x, want %#x", hwcap, hwcapASIMDDP)
	}
	if hwcap2 != 0 {
		t.Fatalf("hwcap2 = %#x, want 0 (entry is past AT_NULL)", hwcap2)
	}
}

func TestParseAuxvHWCapTruncated(t *testing.T) {
	// A trailing partial (<16-byte) entry must be ignored, not read OOB.
	img := append(auxvImage([][2]uint64{{atHWCap, hwcapASIMDDP}}), 0x01, 0x02, 0x03)
	if hwcap, _ := parseAuxvHWCap(img); hwcap != hwcapASIMDDP {
		t.Fatalf("hwcap = %#x, want %#x", hwcap, hwcapASIMDDP)
	}
}

func TestApplyARM64HWCap(t *testing.T) {
	cases := []struct {
		name                     string
		hwcap, hwcap2            uint64
		dotprod, i8mm, sve, sve2 bool
	}{
		{"none", 0, 0, false, false, false, false},
		{"dotprod", hwcapASIMDDP, 0, true, false, false, false},
		{"i8mm", 0, hwcap2I8MM, false, true, false, false},
		{"sve+sve2", hwcapSVE, hwcap2SVE2, false, false, true, true},
		{"all", hwcapASIMDDP | hwcapSVE, hwcap2I8MM | hwcap2SVE2, true, true, true, true},
		// Unrelated bits must not leak into our features.
		{"noise", 1 << 0, 1 << 0, false, false, false, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var f Features
			applyARM64HWCap(&f, c.hwcap, c.hwcap2)
			if f.DOTPROD != c.dotprod || f.I8MM != c.i8mm || f.SVE != c.sve || f.SVE2 != c.sve2 {
				t.Fatalf("apply(%#x,%#x) = {DOTPROD:%v I8MM:%v SVE:%v SVE2:%v}, want {%v %v %v %v}",
					c.hwcap, c.hwcap2, f.DOTPROD, f.I8MM, f.SVE, f.SVE2, c.dotprod, c.i8mm, c.sve, c.sve2)
			}
		})
	}
}
