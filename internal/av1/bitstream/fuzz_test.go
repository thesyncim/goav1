package bitstream

import "testing"

func FuzzReadLEB128(f *testing.F) {
	for _, seed := range [][]byte{
		{0x00},
		{0x7f},
		{0x80, 0x01},
		{0xff, 0xff, 0xff, 0xff, 0x0f},
		{0x80},
		{0xff, 0xff, 0xff, 0xff, 0x10},
	} {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		value, n, err := ReadLEB128(data)
		if err != nil {
			return
		}
		if n <= 0 || n > MaxLEB128Bytes || n > len(data) {
			t.Fatalf("invalid consumed length %d for len=%d", n, len(data))
		}
		var buf [MaxLEB128Bytes]byte
		wrote, err := PutLEB128(buf[:], value)
		if err != nil {
			t.Fatalf("PutLEB128(%d): %v", value, err)
		}
		got, consumed, err := ReadLEB128(buf[:wrote])
		if err != nil {
			t.Fatalf("ReadLEB128(encoded): %v", err)
		}
		if got != value || consumed != wrote {
			t.Fatalf("canonical round trip got value=%d consumed=%d want value=%d consumed=%d", got, consumed, value, wrote)
		}
	})
}
