package parser

import "testing"

func FuzzParseSequenceHeader(f *testing.F) {
	f.Add(reducedStillPictureSequenceHeader())
	f.Add(realtimeSequenceHeader())
	f.Add([]byte{0xe0})
	f.Add([]byte{0x00})

	f.Fuzz(func(t *testing.T, payload []byte) {
		sh, err := ParseSequenceHeader(payload)
		if err != nil {
			return
		}
		if sh.SeqProfile > 2 {
			t.Fatalf("accepted invalid profile %d", sh.SeqProfile)
		}
		if sh.ReducedStillPictureHeader && !sh.StillPicture {
			t.Fatal("accepted reduced_still_picture_header without still_picture")
		}
		if sh.OperatingPointsCount == 0 || sh.OperatingPointsCount > 32 {
			t.Fatalf("bad operating point count %d", sh.OperatingPointsCount)
		}
		if sh.MaxFrameWidth == 0 || sh.MaxFrameHeight == 0 {
			t.Fatalf("bad dimensions %dx%d", sh.MaxFrameWidth, sh.MaxFrameHeight)
		}
	})
}

func FuzzParseFrameHeaderPrefix(f *testing.F) {
	f.Add([]byte{0x12, 0x80})
	f.Add([]byte{0x48, 0xc0})
	f.Add([]byte{0x80})

	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	f.Fuzz(func(t *testing.T, payload []byte) {
		hdr, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		if hdr.BitsRead < 0 || hdr.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d len=%d", hdr.BitsRead, len(payload))
		}
		if hdr.PrimaryRefFrame > PrimaryRefNone {
			t.Fatalf("PrimaryRefFrame=%d", hdr.PrimaryRefFrame)
		}
	})
}
