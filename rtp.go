package goav1

import internalrtp "github.com/thesyncim/goav1/internal/av1/rtp"

// RTPAggregationHeader is the parsed AV1 RTP aggregation header (Z/Y/W/N bits
// and element count) defined by the AOM AV1 RTP payload specification.
type RTPAggregationHeader = internalrtp.AggregationHeader

// RTPElement is a single OBU element inside an AV1 RTP payload, with the
// element length pre-decoded and the payload slice aliasing the input.
type RTPElement = internalrtp.Element

// RTPPayloadIterator walks an AV1 RTP payload element by element without
// allocating.
type RTPPayloadIterator = internalrtp.Iterator

// RTPFragmentReassembler buffers OBU fragments across packets and yields
// complete OBU spans once their trailing fragment arrives.
type RTPFragmentReassembler = internalrtp.FragmentReassembler

// RTPDepacketizer converts a sequence of AV1 RTP payloads into reassembled
// OBU spans inside a caller-owned destination buffer.
type RTPDepacketizer = internalrtp.Depacketizer

// RTPObuSpan describes one OBU emitted by RTPDepacketizer: its offset and
// length inside the caller-owned destination buffer.
type RTPObuSpan = internalrtp.OBUSpan

// RTPPayloadSizeLimits bounds the per-packet payload sizing used by the
// packetizer: minimum and maximum RTP payload bytes, and per-OBU header
// overhead.
type RTPPayloadSizeLimits = internalrtp.PayloadSizeLimits

// RTPPacketizer produces AV1 RTP payloads from a sequence of OBUs using
// caller-owned scratch.
type RTPPacketizer = internalrtp.Packetizer

// RTPPacketizerOBU is one OBU entry in the packetizer scratch: a slice of
// the source payload together with metadata describing fragmentation state.
type RTPPacketizerOBU = internalrtp.PacketizerOBU

// RTPPacketPlan is the plan for one output RTP packet: aggregation header,
// element range, and byte size.
type RTPPacketPlan = internalrtp.PacketPlan

type RTPPacketDependencyDescriptorFlags = internalrtp.PacketDependencyDescriptorFlags

// RTPPacketizerScratchSize reports the sizes of the OBU and packet scratches
// required to packetize a payload under given limits.
type RTPPacketizerScratchSize = internalrtp.PacketizerScratchSize

// RTPFrameOBU describes one OBU contained in an assembled AV1 RTP frame, used
// by AssembleRTPFrame to report span offsets and sizes.
type RTPFrameOBU = internalrtp.FrameOBU

// RTPDependencyDescriptorMandatory is the fixed 3-byte mandatory prefix of
// WebRTC's dependency descriptor RTP header extension.
type RTPDependencyDescriptorMandatory = internalrtp.DependencyDescriptorMandatory

// RTPDependencyDescriptorDecodeTargetIndication describes whether a frame is
// not present, discardable, a switch point, or required for a decode target.
type RTPDependencyDescriptorDecodeTargetIndication = internalrtp.DependencyDescriptorDecodeTargetIndication

// RTPDependencyDescriptorResolution is one render-resolution entry from an
// attached dependency descriptor structure.
type RTPDependencyDescriptorResolution = internalrtp.DependencyDescriptorResolution

// RTPDependencyDescriptorFrameDependencies is the template-derived or custom
// dependency definition for a parsed frame.
type RTPDependencyDescriptorFrameDependencies = internalrtp.DependencyDescriptorFrameDependencies

// RTPDependencyDescriptorStructure is the parsed template dependency
// structure carried by a WebRTC dependency descriptor RTP header extension.
type RTPDependencyDescriptorStructure = internalrtp.DependencyDescriptorStructure

// RTPDependencyDescriptor is the full parsed WebRTC dependency descriptor RTP
// header extension value.
type RTPDependencyDescriptor = internalrtp.DependencyDescriptor

// RTPDependencyDescriptorState remembers the last attached dependency
// structure and current active decode-target mask, then parses compact later
// descriptors against that state.
type RTPDependencyDescriptorState = internalrtp.DependencyDescriptorState

const RTPDependencyDescriptorMandatorySize = internalrtp.DependencyDescriptorMandatorySize

const (
	RTPDependencyDescriptorMaxSpatialLayers  = internalrtp.DependencyDescriptorMaxSpatialLayers
	RTPDependencyDescriptorMaxTemporalLayers = internalrtp.DependencyDescriptorMaxTemporalLayers
	RTPDependencyDescriptorMaxDecodeTargets  = internalrtp.DependencyDescriptorMaxDecodeTargets
	RTPDependencyDescriptorMaxTemplates      = internalrtp.DependencyDescriptorMaxTemplates
	RTPDependencyDescriptorMaxFrameDiffs     = internalrtp.DependencyDescriptorMaxFrameDiffs
)

const (
	RTPDependencyDescriptorDecodeTargetNotPresent  = internalrtp.DependencyDescriptorDecodeTargetNotPresent
	RTPDependencyDescriptorDecodeTargetDiscardable = internalrtp.DependencyDescriptorDecodeTargetDiscardable
	RTPDependencyDescriptorDecodeTargetSwitch      = internalrtp.DependencyDescriptorDecodeTargetSwitch
	RTPDependencyDescriptorDecodeTargetRequired    = internalrtp.DependencyDescriptorDecodeTargetRequired
)

// RTP packetization/depacketization errors. Each Err* value is returned by
// the RTP helpers when their input violates the AV1 RTP payload framing rules
// or the supplied caller buffers are too small.
var (
	ErrRTPShortPayload                = internalrtp.ErrShortPayload
	ErrRTPReservedBit                 = internalrtp.ErrReservedBit
	ErrRTPInvalidAggregationHeader    = internalrtp.ErrInvalidAggregationHeader
	ErrRTPInvalidElementCount         = internalrtp.ErrInvalidElementCount
	ErrRTPShortBuffer                 = internalrtp.ErrShortBuffer
	ErrRTPZeroLengthElement           = internalrtp.ErrZeroLengthElement
	ErrRTPUnexpectedContinuation      = internalrtp.ErrUnexpectedContinuation
	ErrRTPFragmentInterrupted         = internalrtp.ErrFragmentInterrupted
	ErrRTPEmptyFrame                  = internalrtp.ErrEmptyFrame
	ErrRTPMTUTooSmall                 = internalrtp.ErrMTUTooSmall
	ErrRTPInvalidPayloadLimits        = internalrtp.ErrInvalidPayloadLimits
	ErrRTPOBUBufferTooSmall           = internalrtp.ErrOBUBufferTooSmall
	ErrRTPPacketPlanTooSmall          = internalrtp.ErrPacketPlanTooSmall
	ErrRTPInvalidDependencyDescriptor = internalrtp.ErrInvalidDependencyDescriptor
)

// ParseRTPAggregationHeader parses an AV1 RTP aggregation header at the front
// of src and returns the parsed header and the number of bytes consumed.
func ParseRTPAggregationHeader(src []byte) (RTPAggregationHeader, int, error) {
	return internalrtp.ParseAggregationHeader(src)
}

// PutRTPAggregationHeader serializes header into dst and returns the number
// of bytes written. dst must hold at least one byte.
func PutRTPAggregationHeader(dst []byte, header RTPAggregationHeader) (int, error) {
	return internalrtp.PutAggregationHeader(dst, header)
}

// NewRTPPayloadIterator returns an iterator over the OBU elements in an AV1
// RTP payload. The iterator aliases payload.
func NewRTPPayloadIterator(payload []byte) (RTPPayloadIterator, error) {
	return internalrtp.NewIterator(payload)
}

// PutRTPPayload writes an AV1 RTP payload composed of header and elements
// into dst and returns the number of bytes written.
func PutRTPPayload(dst []byte, header RTPAggregationHeader, elements []RTPElement) (int, error) {
	return internalrtp.PutPayload(dst, header, elements)
}

func PutRTPDependencyDescriptorMandatory(dst []byte, descriptor RTPDependencyDescriptorMandatory) (int, error) {
	return internalrtp.PutDependencyDescriptorMandatory(dst, descriptor)
}

func ParseRTPDependencyDescriptorMandatory(src []byte) (RTPDependencyDescriptorMandatory, int, error) {
	return internalrtp.ParseDependencyDescriptorMandatory(src)
}

// ParseRTPDependencyDescriptor parses a full WebRTC dependency descriptor RTP
// header extension. structure may be nil only when src carries an attached
// structure; pass the last attached structure for compact later descriptors.
func ParseRTPDependencyDescriptor(src []byte, structure *RTPDependencyDescriptorStructure) (RTPDependencyDescriptor, int, error) {
	return internalrtp.ParseDependencyDescriptor(src, structure)
}

// PutRTPFragment writes one MTU-sized RTP fragment of an OBU into dst. offset
// is the byte offset of the OBU at which the fragment starts; the function
// returns the number of bytes written, the next offset, and whether more
// fragments are required to encode the OBU. startsNewCodedVideoSequence
// controls the N bit of the aggregation header.
func PutRTPFragment(dst []byte, obu []byte, offset int, mtu int, startsNewCodedVideoSequence bool) (n int, nextOffset int, more bool, err error) {
	return internalrtp.PutFragment(dst, obu, offset, mtu, startsNewCodedVideoSequence)
}

// NewRTPPacketizer initializes a Packetizer that writes RTP payloads into the
// caller-owned obuScratch, packetScratch, and workScratch slices. keyFrame
// and lastFrameInPicture set the corresponding aggregation-header bits.
func NewRTPPacketizer(payload []byte, limits RTPPayloadSizeLimits, keyFrame bool, lastFrameInPicture bool, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (RTPPacketizer, error) {
	return internalrtp.NewPacketizer(payload, limits, keyFrame, lastFrameInPicture, obuScratch, packetScratch, workScratch)
}

func NextRTPPacketDependencyDescriptorFlags(packetizer *RTPPacketizer) (RTPPacketDependencyDescriptorFlags, bool) {
	return packetizer.NextPacketDependencyDescriptorFlags()
}

func RTPPacketizerRemainingPayloadSize(packetizer *RTPPacketizer) int {
	return packetizer.RemainingPacketPayloadSize()
}

// RTPPacketizerScratchLen reports the OBU and packet scratch capacities the
// packetizer requires for payload under the supplied limits. obuScratch can
// be nil; if non-nil it is used to compute the OBU element count.
func RTPPacketizerScratchLen(payload []byte, limits RTPPayloadSizeLimits, obuScratch []RTPPacketizerOBU) (RTPPacketizerScratchSize, error) {
	return internalrtp.PacketizerScratchLen(payload, limits, obuScratch)
}

// AssembleRTPFrame concatenates the supplied AV1 RTP payloads into a complete
// frame inside dst, restoring obu_size fields for each emitted OBU. It
// returns the total bytes written and the number of OBUs emitted.
func AssembleRTPFrame(dst []byte, payloads [][]byte, obus []RTPFrameOBU) (n int, obuCount int, err error) {
	return internalrtp.AssembleFrame(dst, payloads, obus)
}

// AssembleRTPFrameSize reports the number of bytes and OBUs that
// AssembleRTPFrame would write for the supplied payloads, without producing
// the assembled frame.
func AssembleRTPFrameSize(payloads [][]byte) (n int, obuCount int, err error) {
	return internalrtp.AssembleFrameSize(payloads)
}
