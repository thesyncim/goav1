package obu

import "testing"

func FuzzParseHeader(f *testing.F) {
	for _, seed := range [][]byte{
		{byte(TypeSequenceHeader) << 3},
		{byte(TypeFrameHeader)<<3 | 0x06, 0x28},
		{0x80},
		{0x01},
	} {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		h, n, err := ParseHeader(data)
		if err != nil {
			return
		}
		if n != h.HeaderLen() || n <= 0 || n > len(data) {
			t.Fatalf("bad header len n=%d headerLen=%d src=%d", n, h.HeaderLen(), len(data))
		}
		var out [2]byte
		wrote, err := PutHeader(out[:], h)
		if err != nil {
			t.Fatalf("PutHeader(%+v): %v", h, err)
		}
		if wrote != n {
			t.Fatalf("PutHeader wrote=%d parsed=%d", wrote, n)
		}
	})
}
