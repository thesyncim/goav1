package rtp

import "testing"

func FuzzPayloadIterator(f *testing.F) {
	for _, seed := range [][]byte{
		{0x00, 0x01, 0xaa},
		{0x10, 0xaa},
		{0x20, 0x01, 0xaa, 0xbb},
		{0x88, 0xaa},
		{0x01},
	} {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, payload []byte) {
		it, err := NewIterator(payload)
		if err != nil {
			return
		}
		lastIndex := -1
		for {
			elem, ok, err := it.Next()
			if err != nil {
				return
			}
			if !ok {
				return
			}
			if len(elem.Data) == 0 {
				t.Fatal("empty element returned without error")
			}
			if elem.Index < lastIndex {
				t.Fatalf("index regressed from %d to %d", lastIndex, elem.Index)
			}
			lastIndex = elem.Index
		}
	})
}
