package goav1

import (
	"errors"
	"unicode/utf8"
)

const (
	// RTPSDESItemTypeRTPStreamID is the RFC 8852 RTCP SDES item type for
	// RtpStreamId.
	RTPSDESItemTypeRTPStreamID = 12
	// RTPSDESItemTypeRepairedRTPStreamID is the RFC 8852 RTCP SDES item type
	// for RepairedRtpStreamId.
	RTPSDESItemTypeRepairedRTPStreamID = 13
	// RTPSDESItemTypeMID is the RTP MID SDES item type.
	RTPSDESItemTypeMID = 15
	// RTPHeaderExtensionSDESMaxLen is the maximum SDES item text length.
	RTPHeaderExtensionSDESMaxLen = 255
)

// ErrRTPInvalidSDESItem is returned when an RTP SDES header-extension value is
// malformed or outside the relevant RFC length/character constraints.
var ErrRTPInvalidSDESItem = errors.New("goav1: invalid RTP SDES item")

// ValidateRTPMID reports whether mid is valid for the MID RTP header
// extension payload. MID values use SDP token syntax and are limited to the
// SDES item length.
func ValidateRTPMID(mid string) error {
	if len(mid) == 0 || len(mid) > RTPHeaderExtensionSDESMaxLen {
		return ErrRTPInvalidSDESItem
	}
	for i := 0; i < len(mid); i++ {
		if !av1SDPTokenChar(mid[i]) {
			return ErrRTPInvalidSDESItem
		}
	}
	return nil
}

// ParseRTPMIDHeaderExtension parses a MID RTP header-extension element
// payload. The RTP extension element header is not part of src.
func ParseRTPMIDHeaderExtension(src []byte) (string, error) {
	mid := string(src)
	if err := ValidateRTPMID(mid); err != nil {
		return "", err
	}
	return mid, nil
}

// PutRTPMIDHeaderExtension writes a MID RTP header-extension element payload.
// The RTP extension element header is not written.
func PutRTPMIDHeaderExtension(dst []byte, mid string) (int, error) {
	if err := ValidateRTPMID(mid); err != nil {
		return 0, err
	}
	if len(dst) < len(mid) {
		return 0, ErrRTPShortBuffer
	}
	return copy(dst, mid), nil
}

// ValidateRTPStreamID reports whether id is valid for the RtpStreamId or
// RepairedRtpStreamId RTP header extension payload.
func ValidateRTPStreamID(id string) error {
	if len(id) == 0 || len(id) > RTPHeaderExtensionSDESMaxLen {
		return ErrRTPInvalidSDESItem
	}
	for i := 0; i < len(id); i++ {
		if !av1SDPAlphaNumeric(id[i]) {
			return ErrRTPInvalidSDESItem
		}
	}
	return nil
}

// ParseRTPStreamIDHeaderExtension parses an RtpStreamId RTP header-extension
// element payload. The RTP extension element header is not part of src.
func ParseRTPStreamIDHeaderExtension(src []byte) (string, error) {
	id := string(src)
	if err := ValidateRTPStreamID(id); err != nil {
		return "", err
	}
	return id, nil
}

// PutRTPStreamIDHeaderExtension writes an RtpStreamId RTP header-extension
// element payload. The RTP extension element header is not written.
func PutRTPStreamIDHeaderExtension(dst []byte, id string) (int, error) {
	return putRTPStreamIDLikeHeaderExtension(dst, id)
}

// ParseRTPRepairedStreamIDHeaderExtension parses a RepairedRtpStreamId RTP
// header-extension element payload. The RTP extension element header is not
// part of src.
func ParseRTPRepairedStreamIDHeaderExtension(src []byte) (string, error) {
	id := string(src)
	if err := ValidateRTPStreamID(id); err != nil {
		return "", err
	}
	return id, nil
}

// PutRTPRepairedStreamIDHeaderExtension writes a RepairedRtpStreamId RTP
// header-extension element payload. The RTP extension element header is not
// written.
func PutRTPRepairedStreamIDHeaderExtension(dst []byte, id string) (int, error) {
	return putRTPStreamIDLikeHeaderExtension(dst, id)
}

// ValidateRTPSDESSourceDescriptionText reports whether value fits the generic
// RFC 7941 RTP SDES header-extension text-value envelope. Item-specific
// helpers such as ValidateRTPMID and ValidateRTPStreamID apply stricter
// non-empty and character constraints.
func ValidateRTPSDESSourceDescriptionText(value string) error {
	if len(value) > RTPHeaderExtensionSDESMaxLen || !utf8.ValidString(value) {
		return ErrRTPInvalidSDESItem
	}
	return nil
}

// ParseRTPSDESSourceDescriptionTextHeaderExtension parses a generic RFC 7941
// RTP SDES header-extension element payload. The RTP extension element header
// is not part of src.
func ParseRTPSDESSourceDescriptionTextHeaderExtension(src []byte) (string, error) {
	value := string(src)
	if err := ValidateRTPSDESSourceDescriptionText(value); err != nil {
		return "", err
	}
	return value, nil
}

// PutRTPSDESSourceDescriptionTextHeaderExtension writes a generic RFC 7941
// RTP SDES header-extension element payload. The RTP extension element header
// is not written.
func PutRTPSDESSourceDescriptionTextHeaderExtension(
	dst []byte, value string,
) (int, error) {
	if err := ValidateRTPSDESSourceDescriptionText(value); err != nil {
		return 0, err
	}
	if len(dst) < len(value) {
		return 0, ErrRTPShortBuffer
	}
	return copy(dst, value), nil
}

func putRTPStreamIDLikeHeaderExtension(dst []byte, id string) (int, error) {
	if err := ValidateRTPStreamID(id); err != nil {
		return 0, err
	}
	if len(dst) < len(id) {
		return 0, ErrRTPShortBuffer
	}
	return copy(dst, id), nil
}
