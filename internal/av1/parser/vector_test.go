package parser

import (
	"bytes"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/testvector"
)

func TestSequenceHeaderCoreVectors(t *testing.T) {
	suite := testvector.CoreSuite()
	oracle := testvector.NewOracle(suite.Manifest)
	for _, tag := range [...]testvector.Tag{
		testvector.TagParserSequenceFullLowOverhead,
		testvector.TagParserSequenceReducedLowOverhead,
		testvector.TagParserSequenceFullAnnexB,
		testvector.TagParserSequenceReducedAnnexB,
		testvector.TagParserSequenceBuganizer502133197,
	} {
		vector, ok := suite.Manifest.Find(tag)
		if !ok {
			t.Fatalf("missing vector tag=%x", tag)
		}
		payload := sequenceVectorPayload(t, vector)
		if !bytes.Equal(payload, vector.Want) {
			t.Fatalf("%s payload=%x want %x", vector.Name, payload, vector.Want)
		}
		parse := ParseSequenceHeader
		if vector.Tag == testvector.TagParserSequenceBuganizer502133197 {
			parse = PeekSequenceHeader
		}
		sh, err := parse(payload)
		if err != nil {
			t.Fatalf("%s sequence header parse: %v", vector.Name, err)
		}
		if vector.Tag != testvector.TagParserSequenceBuganizer502133197 {
			checkSequenceVectorConfig(t, vector, sh)
		}
		if testvector.OracleEnabled {
			if err := oracle.CheckBytes(vector.Tag, payload); err != nil {
				t.Fatalf("%s oracle err=%v", vector.Name, err)
			}
		}
	}
}

func sequenceVectorPayload(t *testing.T, vector testvector.Vector) []byte {
	t.Helper()
	switch vector.Tag {
	case testvector.TagParserSequenceFullLowOverhead, testvector.TagParserSequenceReducedLowOverhead, testvector.TagParserSequenceBuganizer502133197:
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
	case testvector.TagParserSequenceFullAnnexB, testvector.TagParserSequenceReducedAnnexB:
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
		t.Fatalf("%s unhandled tag=%x", vector.Name, vector.Tag)
		return nil
	}
}

func checkSequenceVectorConfig(t *testing.T, vector testvector.Vector, sh SequenceHeader) {
	t.Helper()
	reduced := vector.Tag == testvector.TagParserSequenceReducedLowOverhead || vector.Tag == testvector.TagParserSequenceReducedAnnexB
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
