package testvector

import (
	"bytes"
	"crypto/md5"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/rtp"
)

func TestCoreSuiteVectors(t *testing.T) {
	suite := CoreSuite()
	if suite.Name != "goav1-core" {
		t.Fatalf("suite name=%q", suite.Name)
	}
	if err := suite.Manifest.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, vector := range suite.Manifest.Vectors {
		switch vector.Kind {
		case KindOBU:
			checkOBUVector(t, vector)
		case KindRTP:
			checkRTPVector(t, vector)
		case KindIVF:
			checkIVFVector(t, vector)
		case KindAnnexB:
			checkAnnexBVector(t, vector)
		case KindParser:
			checkParserVector(t, vector)
		default:
			t.Fatalf("unhandled vector kind=%d", vector.Kind)
		}
	}
}

func TestCoreSuiteTagLookup(t *testing.T) {
	suite := CoreSuite()
	for i, vector := range suite.Manifest.Vectors {
		got, ok := CoreVector(vector.Tag)
		if !ok {
			t.Fatalf("missing vector tag=%x", vector.Tag)
		}
		if got.Tag != vector.Tag || got.Kind != vector.Kind || got.Name != vector.Name ||
			!bytes.Equal(got.Input, vector.Input) || !bytes.Equal(got.Want, vector.Want) {
			t.Fatalf("vector[%d]=%+v want %+v", i, got, vector)
		}
	}
	if got, ok := CoreVector(Tag(0xffff)); ok || got.Tag != 0 || got.Kind != 0 || got.Name != "" || got.Input != nil || got.Want != nil {
		t.Fatalf("invalid vector=%+v ok=%v", got, ok)
	}

	digest, ok := CoreFrameDigest(TagIVFSingleFrameAV1, 0)
	if !ok {
		t.Fatal("missing core digest")
	}
	if digest.MD5 != coreIVFSingleFrameMD5 {
		t.Fatalf("digest=%x want %x", digest.MD5, coreIVFSingleFrameMD5)
	}
	if got, ok := CoreFrameDigest(TagIVFSingleFrameAV1, 1); ok || got != (FrameDigest{}) {
		t.Fatalf("invalid digest=%+v ok=%v", got, ok)
	}
}

func TestCoreSuiteAllocs(t *testing.T) {
	allocs := testing.AllocsPerRun(1000, func() {
		suite := CoreSuite()
		if len(suite.Manifest.Vectors) != len(coreVectors) || len(suite.Manifest.Digests) != len(coreDigests) {
			t.Fatalf("suite=%+v", suite)
		}
		if _, ok := CoreVector(TagParserSequenceFullLowOverhead); !ok {
			t.Fatal("missing vector")
		}
		if _, ok := CoreFrameDigest(TagIVFSingleFrameAV1, 0); !ok {
			t.Fatal("missing digest")
		}
	})
	if allocs != 0 {
		t.Fatalf("CoreSuite allocated: %f", allocs)
	}
}

func checkOBUVector(t *testing.T, vector Vector) {
	t.Helper()
	unit, consumed, err := obu.ParseLowOverhead(vector.Input)
	if err != nil {
		t.Fatalf("%s ParseLowOverhead: %v", vector.Name, err)
	}
	if consumed != len(vector.Input) {
		t.Fatalf("%s consumed=%d want %d", vector.Name, consumed, len(vector.Input))
	}
	if unit.Header.Type != obu.TypeTemporalDelimiter {
		t.Fatalf("%s type=%d want %d", vector.Name, unit.Header.Type, obu.TypeTemporalDelimiter)
	}
	if !bytes.Equal(unit.Payload, vector.Want) {
		t.Fatalf("%s payload=%x want %x", vector.Name, unit.Payload, vector.Want)
	}
}

func checkRTPVector(t *testing.T, vector Vector) {
	t.Helper()
	it, err := rtp.NewIterator(vector.Input)
	if err != nil {
		t.Fatalf("%s NewIterator: %v", vector.Name, err)
	}
	elem, ok, err := it.Next()
	if err != nil {
		t.Fatalf("%s Next: %v", vector.Name, err)
	}
	if !ok {
		t.Fatalf("%s missing element", vector.Name)
	}
	if !bytes.Equal(elem.Data, vector.Want) {
		t.Fatalf("%s data=%x want %x", vector.Name, elem.Data, vector.Want)
	}
}

func checkIVFVector(t *testing.T, vector Vector) {
	t.Helper()
	it, err := ivf.NewIterator(vector.Input)
	if err != nil {
		t.Fatalf("%s NewIterator: %v", vector.Name, err)
	}
	frame, ok, err := it.Next()
	if err != nil {
		t.Fatalf("%s Next: %v", vector.Name, err)
	}
	if !ok {
		t.Fatalf("%s missing frame", vector.Name)
	}
	if !bytes.Equal(frame.Payload, vector.Want) {
		t.Fatalf("%s payload=%x want %x", vector.Name, frame.Payload, vector.Want)
	}
	digest, ok := CoreFrameDigest(vector.Tag, frame.Index)
	if !ok {
		t.Fatalf("%s missing digest", vector.Name)
	}
	if got := MD5(md5.Sum(frame.Payload)); got != digest.MD5 {
		t.Fatalf("%s md5=%x want %x", vector.Name, got, digest.MD5)
	}
}

func checkAnnexBVector(t *testing.T, vector Vector) {
	t.Helper()
	it := obu.NewAnnexBIterator(vector.Input)
	wantOff := 0
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("%s Next: %v", vector.Name, err)
		}
		if !ok {
			break
		}
		end := wantOff + len(unit.Raw)
		if end > len(vector.Want) {
			t.Fatalf("%s raw overrun off=%d len=%d want=%d", vector.Name, wantOff, len(unit.Raw), len(vector.Want))
		}
		if !bytes.Equal(unit.Raw, vector.Want[wantOff:end]) {
			t.Fatalf("%s raw=%x want=%x", vector.Name, unit.Raw, vector.Want[wantOff:end])
		}
		wantOff = end
	}
	if wantOff != len(vector.Want) {
		t.Fatalf("%s consumed raw=%d want %d", vector.Name, wantOff, len(vector.Want))
	}
}

func checkParserVector(t *testing.T, vector Vector) {
	t.Helper()
	payload := sequencePayloadFromVector(t, vector)
	if !bytes.Equal(payload, vector.Want) {
		t.Fatalf("%s payload=%x want %x", vector.Name, payload, vector.Want)
	}
	parse := parser.ParseSequenceHeader
	if vector.Tag == TagParserSequenceBuganizer502133197 {
		parse = parser.PeekSequenceHeader
	}
	sh, err := parse(payload)
	if err != nil {
		t.Fatalf("%s sequence header parse: %v", vector.Name, err)
	}
	if vector.Tag != TagParserSequenceBuganizer502133197 {
		checkLibaomSequenceConfig(t, vector, sh)
	}
}

func sequencePayloadFromVector(t *testing.T, vector Vector) []byte {
	t.Helper()
	switch vector.Tag {
	case TagParserSequenceFullLowOverhead, TagParserSequenceReducedLowOverhead, TagParserSequenceBuganizer502133197:
		unit, consumed, err := obu.ParseLowOverhead(vector.Input)
		if err != nil {
			t.Fatalf("%s ParseLowOverhead: %v", vector.Name, err)
		}
		if consumed != len(vector.Input) {
			t.Fatalf("%s consumed=%d want %d", vector.Name, consumed, len(vector.Input))
		}
		if unit.Header.Type != obu.TypeSequenceHeader {
			t.Fatalf("%s type=%d want %d", vector.Name, unit.Header.Type, obu.TypeSequenceHeader)
		}
		return unit.Payload
	case TagParserSequenceFullAnnexB, TagParserSequenceReducedAnnexB:
		unit, consumed, err := obu.ParseAnnexBElement(vector.Input)
		if err != nil {
			t.Fatalf("%s ParseAnnexBElement: %v", vector.Name, err)
		}
		if consumed != len(vector.Input) {
			t.Fatalf("%s consumed=%d want %d", vector.Name, consumed, len(vector.Input))
		}
		if unit.Header.Type != obu.TypeSequenceHeader {
			t.Fatalf("%s type=%d want %d", vector.Name, unit.Header.Type, obu.TypeSequenceHeader)
		}
		return unit.Payload
	default:
		t.Fatalf("%s unhandled parser vector tag=%x", vector.Name, vector.Tag)
		return nil
	}
}

func checkLibaomSequenceConfig(t *testing.T, vector Vector, sh parser.SequenceHeader) {
	t.Helper()
	reduced := vector.Tag == TagParserSequenceReducedLowOverhead || vector.Tag == TagParserSequenceReducedAnnexB
	if sh.SeqProfile != 0 || sh.OperatingPointsCount != 1 || sh.OperatingPoints[0].SeqLevelIdx != 0 || sh.OperatingPoints[0].SeqTier != 0 {
		t.Fatalf("%s profile/op=%+v", vector.Name, sh)
	}
	if sh.StillPicture != reduced || sh.ReducedStillPictureHeader != reduced {
		t.Fatalf("%s still/reduced=%v/%v want %v", vector.Name, sh.StillPicture, sh.ReducedStillPictureHeader, reduced)
	}
	if sh.InitialDisplayDelayPresent || sh.OperatingPoints[0].InitialDisplayDelayPresent || sh.OperatingPoints[0].InitialDisplayDelayMinus1 != 0 {
		t.Fatalf("%s initial delay=%+v", vector.Name, sh.OperatingPoints[0])
	}
	cc := sh.ColorConfig
	if cc.HighBitdepth || cc.TwelveBit || cc.MonoChrome || !cc.SubsamplingX || !cc.SubsamplingY || cc.ChromaSamplePosition != 0 {
		t.Fatalf("%s color config=%+v", vector.Name, cc)
	}
}
