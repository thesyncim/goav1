package testvector

import "testing"

func TestLibaomRemoteManifest(t *testing.T) {
	manifest := LibaomRemoteManifest()
	if err := manifest.Validate(); err != nil {
		t.Fatal(err)
	}
	if manifest.BaseURL != LibaomTestDataURL {
		t.Fatalf("base url=%q want %q", manifest.BaseURL, LibaomTestDataURL)
	}
	if len(manifest.Vectors) != len(libaomRemoteVectors) {
		t.Fatalf("vectors=%d want %d", len(manifest.Vectors), len(libaomRemoteVectors))
	}
	for _, vector := range manifest.Vectors {
		if vector.Stream.Name == "" || vector.MD5.Name != vector.Stream.Name+".md5" {
			t.Fatalf("vector files=%+v", vector)
		}
		if vector.Oracle != OracleFrameMD5 {
			t.Fatalf("%s oracle=%d", vector.Name, vector.Oracle)
		}
	}
}

func TestLibaomRemoteVectorLookupAndLabels(t *testing.T) {
	vector, ok := LibaomRemoteVector(TagDecoderLibaomQuantizer00)
	if !ok {
		t.Fatal("missing quantizer vector")
	}
	if vector.Stream.Name != "av1-1-b8-00-quantizer-00.ivf" || vector.Labels&VectorLabelQuantizer == 0 {
		t.Fatalf("quantizer vector=%+v", vector)
	}
	if got, ok := LibaomRemoteVector(Tag(0xffff)); ok || got.Tag != 0 {
		t.Fatalf("invalid vector=%+v ok=%v", got, ok)
	}

	manifest := LibaomRemoteManifest()
	fast := manifest.SelectRemote(SuiteLevelFast, 0, nil)
	if len(fast) != 14 {
		t.Fatalf("fast vectors=%d want 14", len(fast))
	}
	motion := manifest.SelectRemote(SuiteLevelRelevant, VectorLabelMotion, nil)
	if len(motion) != 2 {
		t.Fatalf("motion vectors=%d want 2", len(motion))
	}
	highBitDepth := manifest.SelectRemote(SuiteLevelRelevant, VectorLabelHighBitDepth, nil)
	if len(highBitDepth) != 3 {
		t.Fatalf("high-bit-depth vectors=%d want 3", len(highBitDepth))
	}
	full := manifest.SelectRemote(SuiteLevelFull, 0, nil)
	if len(full) != len(libaomRemoteVectors) {
		t.Fatalf("full vectors=%d want %d", len(full), len(libaomRemoteVectors))
	}
	extended := manifest.SelectRemote(SuiteLevelExtended, 0, nil)
	if len(extended) != 226 {
		t.Fatalf("extended vectors=%d want 226", len(extended))
	}
	// Extended cohort must be opt-in: every extended vector also has
	// SuiteLevelFull set, but none of them carries SuiteLevelFast or
	// SuiteLevelRelevant.
	for _, vector := range extended {
		if vector.Levels&(SuiteLevelFast|SuiteLevelRelevant) != 0 {
			t.Fatalf("extended vector %q leaks into default tiers: levels=%b", vector.Name, vector.Levels)
		}
		if vector.Levels&SuiteLevelFull == 0 {
			t.Fatalf("extended vector %q missing SuiteLevelFull: levels=%b", vector.Name, vector.Levels)
		}
	}
	extendedHighBitDepth := manifest.SelectRemote(SuiteLevelExtended, VectorLabelHighBitDepth, nil)
	if len(extendedHighBitDepth) != 63 {
		t.Fatalf("extended high-bit-depth vectors=%d want 63", len(extendedHighBitDepth))
	}
	extendedSize := manifest.SelectRemote(SuiteLevelExtended, VectorLabelSize, nil)
	if len(extendedSize) != 99 {
		t.Fatalf("extended size vectors=%d want 99", len(extendedSize))
	}
	extendedSVC := manifest.SelectRemote(SuiteLevelExtended, VectorLabelSVC, nil)
	if len(extendedSVC) != 2 {
		t.Fatalf("extended svc vectors=%d want 2", len(extendedSVC))
	}
}

func TestParseLibaomMD5Digests(t *testing.T) {
	digests, err := ParseLibaomMD5Digests(TagDecoderLibaomQuantizer00, []byte(
		"5a68de997d60afa9083b17fe00f7cdf2  frame0\n"+
			"00112233445566778899aabbccddeeff  frame1\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(digests) != 2 {
		t.Fatalf("digests=%d want 2", len(digests))
	}
	if digests[0].FrameIndex != 0 || digests[1].FrameIndex != 1 {
		t.Fatalf("digests=%+v", digests)
	}
	if _, err := ParseLibaomMD5Digests(TagDecoderLibaomQuantizer00, nil); err != ErrMissingDigest {
		t.Fatalf("empty err=%v want %v", err, ErrMissingDigest)
	}
	if _, err := ParseLibaomMD5Digests(TagDecoderLibaomQuantizer00, []byte("bad")); err == nil {
		t.Fatal("bad md5 err=nil")
	}
}
