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

func FuzzParseIntraFrameSize(f *testing.F) {
	f.Add([]byte{0x10, 0x00})
	f.Add([]byte{0x08, 0x00})
	f.Add([]byte{0x04, 0x00})

	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseIntraFrameSize(payload, seq, prefix, 0, 0)
		if err != nil {
			return
		}
		if size.BitsRead < prefix.BitsRead || size.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d prefix=%d len=%d", size.BitsRead, prefix.BitsRead, len(payload))
		}
		if size.CodedWidth == 0 || size.UpscaledWidth == 0 || size.Height == 0 {
			t.Fatalf("bad dimensions=%+v", size)
		}
		if size.CodedWidth > size.UpscaledWidth {
			t.Fatalf("coded width=%d upscaled=%d", size.CodedWidth, size.UpscaledWidth)
		}
	})
}

func FuzzParseFrameSize(f *testing.F) {
	f.Add([]byte{0x10, 0x00})
	f.Add([]byte{0x12, 0x00, 0x00, 0x00, 0x00})
	f.Add([]byte{0x09, 0x00, 0x00, 0x00, 0x00})

	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}
	var refs ReferenceState
	for i := 0; i < RefFrames; i++ {
		refs.Frames[i] = ReferenceFrame{
			Valid:     true,
			OrderHint: uint32(i),
			Size: FrameSize{
				CodedWidth:          seq.MaxFrameWidth,
				UpscaledWidth:       seq.MaxFrameWidth,
				Height:              seq.MaxFrameHeight,
				RenderWidth:         seq.MaxFrameWidth,
				RenderHeight:        seq.MaxFrameHeight,
				SuperResDenominator: 8,
			},
		}
	}

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
		if err != nil {
			return
		}
		if size.BitsRead < prefix.BitsRead || size.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d prefix=%d len=%d", size.BitsRead, prefix.BitsRead, len(payload))
		}
		if size.CodedWidth == 0 || size.UpscaledWidth == 0 || size.Height == 0 {
			t.Fatalf("bad dimensions=%+v", size)
		}
	})
}
