package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/pion/rtcp"
	"github.com/pion/rtp"
	"github.com/pion/rtp/codecs"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media/samplebuilder"

	goav1 "github.com/thesyncim/goav1"
)

// TestEndToEndAV1OverRTP proves the whole media path without a browser: a
// receiving peer connection (standing in for the browser) negotiates with
// the sender, the encoder's temporal units travel as AV1 RTP, and the
// depacketized stream must decode in the goav1 decoder with real picture
// content - both ends implement the same RTP and bitstream specs the
// browser does.
func TestEndToEndAV1OverRTP(t *testing.T) {
	receiver, decoded, _ := newAV1ReceivingPeer(t)
	defer receiver.Close()

	sender, err := webrtc.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	defer sender.Close()
	waitSenderConnected := waitPeerConnected(t, sender, "sender")
	track, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeAV1}, "video", "goav1")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sender.AddTrack(track); err != nil {
		t.Fatal(err)
	}

	// In-process signaling.
	offer, err := receiver.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	og := webrtc.GatheringCompletePromise(receiver)
	if err := receiver.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	<-og
	if err := sender.SetRemoteDescription(*receiver.LocalDescription()); err != nil {
		t.Fatal(err)
	}
	answer, err := sender.CreateAnswer(nil)
	if err != nil {
		t.Fatal(err)
	}
	ag := webrtc.GatheringCompletePromise(sender)
	if err := sender.SetLocalDescription(answer); err != nil {
		t.Fatal(err)
	}
	<-ag
	if err := receiver.SetRemoteDescription(*sender.LocalDescription()); err != nil {
		t.Fatal(err)
	}
	waitSenderConnected()

	// Stream a couple of seconds through the production path.
	var wantKey atomic.Bool
	wantKey.Store(true)
	done := make(chan struct{})
	go func() {
		stream(track, &wantKey, 4_000_000, done)
	}()
	defer close(done)
	// The browser answers early loss with picture-loss feedback until a
	// keyframe lands; the test stands in for that loop.
	go func() {
		t := time.NewTicker(500 * time.Millisecond)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				wantKey.Store(true)
			}
		}
	}()

	// Collect reassembled temporal units and decode them.
	tus := collectTemporalUnits(t, decoded, 60)
	assertTemporalUnitsDecodeAndReference(t, "browser-push", tus)
}

func TestEndToEndAV1OverRTPOfferEndpoint(t *testing.T) {
	receiver, decoded, trackSSRC := newAV1ReceivingPeer(t)
	defer receiver.Close()
	waitReceiverConnected := waitPeerConnected(t, receiver, "receiver")

	offer, err := receiver.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	og := webrtc.GatheringCompletePromise(receiver)
	if err := receiver.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	<-og

	var serverPC *webrtc.PeerConnection
	req := httptest.NewRequest(http.MethodPost, "/offer", strings.NewReader(receiver.LocalDescription().SDP))
	res := httptest.NewRecorder()
	if err := handleOfferWithPeerConnectionHook(res, req, 4_000_000, func(pc *webrtc.PeerConnection) {
		serverPC = pc
	}); err != nil {
		t.Fatal(err)
	}
	if serverPC == nil {
		t.Fatal("offer handler did not create a peer connection")
	}
	defer serverPC.Close()
	if res.Code != http.StatusOK {
		t.Fatalf("offer handler status=%d body=%q", res.Code, res.Body.String())
	}
	if err := receiver.SetRemoteDescription(webrtc.SessionDescription{
		Type: webrtc.SDPTypeAnswer,
		SDP:  res.Body.String(),
	}); err != nil {
		t.Fatal(err)
	}
	waitReceiverConnected()

	doneFeedback := make(chan struct{})
	defer close(doneFeedback)
	startPictureLossFeedback(t, receiver, trackSSRC, doneFeedback)

	tus := collectTemporalUnits(t, decoded, 60)
	if sequenceHeaders := countSequenceHeaderTemporalUnits(tus); sequenceHeaders < 2 {
		t.Fatalf("sequence headers after PLI=%d want at least 2", sequenceHeaders)
	}
	assertTemporalUnitsDecodeAndReference(t, "browser-push-offer", tus)
}

func TestEndToEndAV1OverRTPRTCEncoderControlChurn(t *testing.T) {
	receiver, decoded, trackSSRC := newAV1ReceivingPeer(t)
	defer receiver.Close()

	sender, err := webrtc.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	defer sender.Close()
	waitSenderConnected := waitPeerConnected(t, sender, "sender")
	track, err := webrtc.NewTrackLocalStaticRTP(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeAV1}, "video", "goav1")
	if err != nil {
		t.Fatal(err)
	}
	rtpSender, err := sender.AddTrack(track)
	if err != nil {
		t.Fatal(err)
	}

	offer, err := receiver.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	og := webrtc.GatheringCompletePromise(receiver)
	if err := receiver.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	<-og
	if err := sender.SetRemoteDescription(*receiver.LocalDescription()); err != nil {
		t.Fatal(err)
	}
	answer, err := sender.CreateAnswer(nil)
	if err != nil {
		t.Fatal(err)
	}
	ag := webrtc.GatheringCompletePromise(sender)
	if err := sender.SetLocalDescription(answer); err != nil {
		t.Fatal(err)
	}
	<-ag
	if err := receiver.SetRemoteDescription(*sender.LocalDescription()); err != nil {
		t.Fatal(err)
	}
	waitSenderConnected()

	done := make(chan struct{})
	defer close(done)
	doneFeedback := make(chan struct{})
	defer close(doneFeedback)
	var wantKey atomic.Bool
	wantKey.Store(true)
	startSenderPictureLossReader(rtpSender, &wantKey, done)
	streamErr := make(chan error, 1)
	go func() {
		streamErr <- streamRTCEncoderRTP(track, &wantKey, done)
	}()
	startPictureLossFeedback(t, receiver, trackSSRC, doneFeedback)

	tus := collectTemporalUnits(t, decoded, 70)
	if sequenceHeaders := countSequenceHeaderTemporalUnits(tus); sequenceHeaders < 3 {
		t.Fatalf("sequence headers after control churn/loss=%d want at least 3", sequenceHeaders)
	}
	select {
	case err := <-streamErr:
		if err != nil {
			t.Fatalf("RTCEncoder RTP stream stopped early: %v", err)
		}
	default:
	}
	assertTemporalUnitsDecodeAndReference(t, "browser-push-rtc-control-churn", tus)
}

type receivedTemporalUnit struct {
	data []byte
}

func newAV1ReceivingPeer(t *testing.T) (*webrtc.PeerConnection, <-chan receivedTemporalUnit, <-chan uint32) {
	t.Helper()
	receiver, err := webrtc.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := receiver.AddTransceiverFromKind(webrtc.RTPCodecTypeVideo, webrtc.RTPTransceiverInit{
		Direction: webrtc.RTPTransceiverDirectionRecvonly,
	}); err != nil {
		receiver.Close()
		t.Fatal(err)
	}

	decoded := make(chan receivedTemporalUnit, 256)
	trackSSRC := make(chan uint32, 1)
	receiver.OnTrack(func(track *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		if track.Codec().MimeType != webrtc.MimeTypeAV1 {
			t.Errorf("negotiated %s, want AV1", track.Codec().MimeType)
			return
		}
		select {
		case trackSSRC <- uint32(track.SSRC()):
		default:
		}
		sb := samplebuilder.New(1024, &codecs.AV1Depacketizer{}, track.Codec().ClockRate)
		for {
			pkt, _, err := track.ReadRTP()
			if err != nil {
				return
			}
			sb.Push(pkt)
			for s := sb.Pop(); s != nil; s = sb.Pop() {
				decoded <- receivedTemporalUnit{data: s.Data}
			}
		}
	})
	return receiver, decoded, trackSSRC
}

func waitPeerConnected(t *testing.T, pc *webrtc.PeerConnection, label string) func() {
	t.Helper()
	connected := make(chan struct{})
	var once sync.Once
	pc.OnConnectionStateChange(func(s webrtc.PeerConnectionState) {
		if s == webrtc.PeerConnectionStateConnected {
			once.Do(func() { close(connected) })
		}
	})
	return func() {
		t.Helper()
		select {
		case <-connected:
		case <-time.After(10 * time.Second):
			t.Fatalf("%s peer connection never connected", label)
		}
	}
}

func startPictureLossFeedback(
	t *testing.T, pc *webrtc.PeerConnection, trackSSRC <-chan uint32, done <-chan struct{},
) {
	t.Helper()
	var mediaSSRC uint32
	select {
	case mediaSSRC = <-trackSSRC:
	case <-time.After(10 * time.Second):
		t.Fatal("remote AV1 track never arrived")
	}
	go func() {
		ticker := time.NewTicker(250 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				if err := pc.WriteRTCP([]rtcp.Packet{
					&rtcp.PictureLossIndication{MediaSSRC: mediaSSRC},
				}); err != nil {
					return
				}
			}
		}
	}()
}

func startSenderPictureLossReader(sender *webrtc.RTPSender, wantKey *atomic.Bool, done <-chan struct{}) {
	startSenderFeedbackReader(sender, wantKey, done, nil)
}

type rtcSenderFeedbackOptions struct {
	Counters *rtcSenderFeedbackCounters
	OnNACK   func(*webrtc.TrackLocalStaticRTP, *rtcp.TransportLayerNack) bool
}

type rtcSenderFeedbackCounters struct {
	PictureLoss atomic.Int64
	FullIntra   atomic.Int64
	NACK        atomic.Int64
}

func startSenderFeedbackReader(
	sender *webrtc.RTPSender, wantKey *atomic.Bool, done <-chan struct{}, counters *rtcSenderFeedbackCounters,
) {
	startSenderFeedbackReaderWithOptions(sender, nil, wantKey, done, rtcSenderFeedbackOptions{Counters: counters})
}

func startSenderFeedbackReaderWithOptions(
	sender *webrtc.RTPSender, track *webrtc.TrackLocalStaticRTP, wantKey *atomic.Bool, done <-chan struct{},
	options rtcSenderFeedbackOptions,
) {
	go func() {
		buf := make([]byte, 1500)
		for {
			select {
			case <-done:
				return
			default:
			}
			n, _, err := sender.Read(buf)
			if err != nil {
				return
			}
			packets, err := rtcp.Unmarshal(buf[:n])
			if err != nil {
				continue
			}
			for _, p := range packets {
				switch feedback := p.(type) {
				case *rtcp.PictureLossIndication:
					if options.Counters != nil {
						options.Counters.PictureLoss.Add(1)
					}
					wantKey.Store(true)
				case *rtcp.FullIntraRequest:
					if options.Counters != nil {
						options.Counters.FullIntra.Add(1)
					}
					wantKey.Store(true)
				case *rtcp.TransportLayerNack:
					if options.Counters != nil {
						options.Counters.NACK.Add(1)
					}
					if options.OnNACK == nil || !options.OnNACK(track, feedback) {
						wantKey.Store(true)
					}
				}
			}
		}
	}()
}

type rtcRTPHeaderExtensions struct {
	DependencyDescriptorID uint8
	TransportWideCCID      uint8
	TransportWideCC02ID    uint8
	Profile                uint16
}

func defaultRTCRTPHeaderExtensions() rtcRTPHeaderExtensions {
	return rtcRTPHeaderExtensions{
		DependencyDescriptorID: 4,
		TransportWideCCID:      5,
		Profile:                goav1.RTPExtensionProfileTwoByte,
	}
}

func streamRTCEncoderRTP(track *webrtc.TrackLocalStaticRTP, wantKey *atomic.Bool, done <-chan struct{}) error {
	return streamRTCEncoderRTPWithHeaderExtensions(track, wantKey, done, defaultRTCRTPHeaderExtensions())
}

func streamRTCEncoderRTPWithHeaderExtensions(
	track *webrtc.TrackLocalStaticRTP, wantKey *atomic.Bool, done <-chan struct{}, headerExtensions rtcRTPHeaderExtensions,
) error {
	options := defaultRTCEncoderRTPStreamOptions()
	options.HeaderExtensions = headerExtensions
	return streamRTCEncoderRTPWithOptions(track, wantKey, done, options)
}

type rtcEncoderRTPStreamOptions struct {
	HeaderExtensions rtcRTPHeaderExtensions
	ForceKeyFrame    func(frameIndex int) bool
	DropPacket       func(frameIndex int, packetIndex int, packet rtp.Packet) bool
	OnPacket         func(frameIndex int, packetIndex int, packet rtp.Packet, dropped bool) error
	OnPicture        func(frameIndex int, picture goav1.RTCPicture)
}

func defaultRTCEncoderRTPStreamOptions() rtcEncoderRTPStreamOptions {
	return rtcEncoderRTPStreamOptions{
		HeaderExtensions: defaultRTCRTPHeaderExtensions(),
		ForceKeyFrame: func(frameIndex int) bool {
			return frameIndex == 28 || frameIndex == 56
		},
	}
}

func streamRTCEncoderRTPWithOptions(
	track *webrtc.TrackLocalStaticRTP, wantKey *atomic.Bool, done <-chan struct{}, options rtcEncoderRTPStreamOptions,
) error {
	if options.HeaderExtensions.Profile == 0 {
		options.HeaderExtensions.Profile = goav1.RTPExtensionProfileTwoByte
	}
	cfg := rtcControlChurnConfig(0)
	enc, err := goav1.NewRTCEncoderWithConfig(cfg)
	if err != nil {
		return err
	}
	defer enc.Close()
	scene := newScene()
	ticker := time.NewTicker(time.Second / fps)
	defer ticker.Stop()
	limits := goav1.RTPPayloadSizeLimits{MaxPayloadLen: 1000}
	var sequence uint16 = 44000
	var timestamp uint32 = 900000
	var timestampRemainder int64
	var twcc uint16 = 1200
	for n := 0; ; n++ {
		select {
		case <-done:
			return nil
		case <-ticker.C:
		}
		if n > 0 && n%14 == 0 {
			cfg = rtcControlChurnConfig(n / 14)
			if err := enc.SetConfig(cfg); err != nil {
				return err
			}
		}
		forceKey := wantKey.Swap(false)
		if options.ForceKeyFrame != nil && options.ForceKeyFrame(n) {
			forceKey = true
		}
		picture, err := enc.EncodePicture(scene.frame(n), forceKey)
		if err != nil {
			return err
		}
		if options.OnPicture != nil {
			options.OnPicture(n, picture)
		}
		packets, nextSequence, nextTWCC, err := rtcPictureRTPPackets(
			picture, limits, sequence, timestamp, twcc, options.HeaderExtensions)
		if err != nil {
			return err
		}
		sequence, twcc = nextSequence, nextTWCC
		for i := range packets {
			dropped := options.DropPacket != nil && options.DropPacket(n, i, packets[i])
			if options.OnPacket != nil {
				if err := options.OnPacket(n, i, packets[i], dropped); err != nil {
					return err
				}
			}
			if dropped {
				continue
			}
			if err := track.WriteRTP(&packets[i]); err != nil {
				return err
			}
		}
		duration, err := enc.RTPFrameDuration()
		if err != nil {
			return err
		}
		timestamp += rtcTimestampIncrement(duration, &timestampRemainder)
	}
}

func rtcTimestampIncrement(duration goav1.EncoderRational, remainder *int64) uint32 {
	total := int64(duration.Num) + *remainder
	step := total / int64(duration.Den)
	*remainder = total % int64(duration.Den)
	return uint32(step)
}

func rtcControlChurnConfig(step int) goav1.EncoderConfig {
	fpsCycle := []goav1.EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60000, Den: 1001},
		{Num: 24, Den: 1},
		{Num: 60, Den: 1},
	}
	target := int32(3600 + (step%4)*450)
	cfg := goav1.EncoderConfig{
		Resolution:         goav1.EncoderResolution{Width: width, Height: height},
		Profile:            goav1.EncoderProfile0,
		BitDepth:           8,
		MaxFramerate:       fpsCycle[step%len(fpsCycle)],
		MinBitrateKbps:     target / 3,
		TargetBitrateKbps:  target,
		MaxBitrateKbps:     target + target/2,
		RateControl:        goav1.EncoderRateControlCBR,
		Content:            goav1.EncoderContentCamera,
		Scalability:        goav1.EncoderScalabilityModeL1T3,
		MaxThreads:         4,
		RTPPacketization:   true,
		LowOverheadOBU:     true,
		DependencyMetadata: true,
	}
	if step%3 == 2 {
		cfg.RateControl = goav1.EncoderRateControlCQP
		cfg.Quantizer = uint8(32 + step%12)
		cfg.Content = goav1.EncoderContentScreen
	} else if step%2 == 1 {
		cfg.Content = goav1.EncoderContentScreen
	}
	return cfg
}

func rtcPictureRTPPackets(
	picture goav1.RTCPicture, limits goav1.RTPPayloadSizeLimits, sequence uint16, timestamp uint32, twcc uint16,
	headerExtensions rtcRTPHeaderExtensions,
) ([]rtp.Packet, uint16, uint16, error) {
	var packets []rtp.Packet
	for i := 0; i < picture.FrameNum; i++ {
		framePackets, nextSequence, nextTWCC, err := rtcFrameRTPPackets(
			picture.Frames[i], limits, sequence, timestamp, twcc, headerExtensions)
		if err != nil {
			return nil, sequence, twcc, err
		}
		sequence, twcc = nextSequence, nextTWCC
		packets = append(packets, framePackets...)
	}
	return packets, sequence, twcc, nil
}

func rtcFrameRTPPackets(
	frame goav1.RTCFrame, limits goav1.RTPPayloadSizeLimits, sequence uint16, timestamp uint32, twcc uint16,
	headerExtensions rtcRTPHeaderExtensions,
) ([]rtp.Packet, uint16, uint16, error) {
	firstSize, err := frame.RTPPacketScratchLen(limits, nil)
	if err != nil {
		return nil, sequence, twcc, err
	}
	obuScratch := make([]goav1.RTPPacketizerOBU, firstSize.Packetizer.OBUs)
	size, err := frame.RTPPacketScratchLen(limits, obuScratch)
	if err != nil {
		return nil, sequence, twcc, err
	}
	packetScratch := make([]goav1.RTPPacketPlan, size.Packetizer.Packets)
	workScratch := make([]goav1.RTPPacketPlan, size.Packetizer.Work)
	spans := make([]goav1.EncoderWebRTCRTPPacketSpan, size.Packetizer.Packets)
	payloads, descriptors, count, err := frame.AppendRTPPackets(
		make([]byte, 0, size.MaxPayloadBytes*size.Packetizer.Packets),
		make([]byte, 0, size.MaxDescriptorBytes*size.Packetizer.Packets),
		spans, limits, obuScratch, packetScratch, workScratch)
	if err != nil {
		return nil, sequence, twcc, err
	}
	if headerExtensions.DependencyDescriptorID == 0 {
		return rtcFrameRTPPacketsWithoutHeaderExtensions(payloads, spans[:count], sequence, timestamp)
	}
	headerConfig := goav1.EncoderWebRTCRTPPacketHeaderConfig{
		PayloadType:                     96,
		SequenceNumber:                  sequence,
		Timestamp:                       timestamp,
		SSRC:                            0xdec0de,
		DependencyDescriptorExtensionID: headerExtensions.DependencyDescriptorID,
		TransportWideCCExtensionID:      headerExtensions.TransportWideCCID,
		TransportWideCC02ExtensionID:    headerExtensions.TransportWideCC02ID,
		TransportWideCCSequenceNumber:   twcc,
		HeaderExtensionProfile:          headerExtensions.Profile,
	}
	headerSize, err := goav1.EncoderWebRTCRTPPacketsWithHeadersSize(
		headerConfig, payloads, descriptors, spans[:count])
	if err != nil {
		return nil, sequence, twcc, err
	}
	headerSpans := make([]goav1.EncoderWebRTCRTPPacketHeaderSpan, count)
	rawPackets, packetCount, err := goav1.AppendEncoderWebRTCRTPPacketsWithHeaders(
		make([]byte, 0, headerSize.Bytes),
		headerSpans,
		headerConfig,
		payloads,
		descriptors,
		spans[:count],
	)
	if err != nil {
		return nil, sequence, twcc, err
	}
	out := make([]rtp.Packet, packetCount)
	for i := range out {
		span := headerSpans[i]
		if err := out[i].Unmarshal(rawPackets[span.Offset : span.Offset+span.Length]); err != nil {
			return nil, sequence, twcc, err
		}
	}
	return out, sequence + uint16(packetCount), twcc + uint16(packetCount), nil
}

func rtcFrameRTPPacketsWithoutHeaderExtensions(
	payloads []byte, spans []goav1.EncoderWebRTCRTPPacketSpan, sequence uint16, timestamp uint32,
) ([]rtp.Packet, uint16, uint16, error) {
	out := make([]rtp.Packet, len(spans))
	for i, span := range spans {
		if span.PayloadOffset < 0 || span.PayloadLength < 0 ||
			span.PayloadOffset > len(payloads) || span.PayloadLength > len(payloads)-span.PayloadOffset {
			return nil, sequence, 0, goav1.ErrEncoderInvalidFrame
		}
		payload := payloads[span.PayloadOffset : span.PayloadOffset+span.PayloadLength]
		out[i] = rtp.Packet{
			Header: rtp.Header{
				Version:        2,
				Marker:         span.Marker,
				PayloadType:    96,
				SequenceNumber: sequence + uint16(i),
				Timestamp:      timestamp,
				SSRC:           0xdec0de,
			},
			Payload: payload,
		}
	}
	return out, sequence + uint16(len(out)), 0, nil
}

func collectTemporalUnits(t *testing.T, decoded <-chan receivedTemporalUnit, want int) [][]byte {
	t.Helper()
	var tus [][]byte
	deadline := time.After(15 * time.Second)
	for len(tus) < want {
		select {
		case u := <-decoded:
			tus = append(tus, u.data)
		case <-deadline:
			t.Fatalf("only %d temporal units arrived", len(tus))
		}
	}
	return tus
}

func assertTemporalUnitsDecodeAndReference(t *testing.T, name string, tus [][]byte) {
	t.Helper()
	start := -1
	for i, u := range tus {
		if len(u) > 0 && (u[0]>>3)&0xF == 1 { // OBU_SEQUENCE_HEADER
			start = i
			break
		}
	}
	if start < 0 {
		t.Fatal("no temporal unit carries a sequence header")
	}
	tus = tus[start:]
	dec, err := goav1.NewDecoder(tus)
	if err != nil {
		t.Fatalf("decoder rejected the depacketized stream: %v", err)
	}
	defer dec.Close()
	frames := 0
	var lumaSum int64
	var lumaCount int64
	var wantYUV []byte
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode frame %d: %v", frames, err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			wantYUV = appendFrameRawYUV(wantYUV, f)
			for y := 0; y < f.Y.Height; y += 64 {
				row := f.Y.Pix[y*f.Y.Stride:]
				for x := 0; x < f.Y.Width; x += 64 {
					lumaSum += int64(row[x])
					lumaCount++
				}
			}
			frames++
		}
	}
	if frames < 25 {
		t.Fatalf("decoded %d frames, want at least 25", frames)
	}
	if lumaCount == 0 {
		t.Fatal("decoded frames had no luma samples")
	}
	avg := lumaSum / lumaCount
	if avg < 40 || avg > 220 {
		t.Fatalf("average luma %d looks like a blank picture", avg)
	}
	ivfFrames := make([]referenceIVFFrame, 0, len(tus))
	for i, payload := range tus {
		ivfFrames = append(ivfFrames, referenceIVFFrame{
			timestamp: uint64(i),
			payload:   payload,
		})
	}
	assertReferenceDecodersRawYUVBytes(
		t,
		referenceAV1Decoders(t),
		name,
		appendReferenceIVF(nil, width, height, fps, 1, ivfFrames),
		wantYUV,
		frames,
	)
	t.Logf("decoded %d frames, average sampled luma %d", frames, avg)
}

func countSequenceHeaderTemporalUnits(tus [][]byte) int {
	count := 0
	for _, u := range tus {
		if len(u) > 0 && (u[0]>>3)&0xF == 1 { // OBU_SEQUENCE_HEADER
			count++
		}
	}
	return count
}

type referenceAV1Decoder struct {
	name string
	path string
	args func(outPath string, ivfPath string) []string
}

func referenceAV1Decoders(t *testing.T) []referenceAV1Decoder {
	t.Helper()
	requireAll := os.Getenv("GOAV1_REQUIRE_WEBRTC_REFERENCE_DECODERS") == "1"
	candidates := []referenceAV1Decoder{
		{
			name: "aomdec",
			args: func(outPath string, ivfPath string) []string {
				return []string{"--rawvideo", "--all-layers", "-o", outPath, ivfPath}
			},
		},
		{
			name: "dav1d",
			args: func(outPath string, ivfPath string) []string {
				return []string{"--alllayers", "1", "--muxer", "yuv", "-o", outPath, "-i", ivfPath}
			},
		},
	}
	decoders := make([]referenceAV1Decoder, 0, len(candidates))
	missing := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		path, err := exec.LookPath(candidate.name)
		if err != nil {
			missing = append(missing, candidate.name)
			t.Logf("%s not on PATH", candidate.name)
			continue
		}
		candidate.path = path
		decoders = append(decoders, candidate)
	}
	if requireAll && len(missing) > 0 {
		t.Fatalf("required reference AV1 decoder(s) not on PATH: %s", strings.Join(missing, ", "))
	}
	if len(decoders) == 0 {
		t.Log("no reference AV1 decoder on PATH; checked in-process WebRTC/decode path only")
	}
	return decoders
}

func assertReferenceDecodersRawYUVBytes(
	t *testing.T, decoders []referenceAV1Decoder, name string, ivf []byte, want []byte, frameCount int,
) {
	t.Helper()
	if len(decoders) == 0 {
		return
	}
	dir := t.TempDir()
	ivfPath := filepath.Join(dir, name+".ivf")
	if err := os.WriteFile(ivfPath, ivf, 0o644); err != nil {
		t.Fatalf("%s write IVF: %v", name, err)
	}
	for _, decoder := range decoders {
		outPath := filepath.Join(dir, name+"-"+decoder.name+".yuv")
		out, err := exec.Command(decoder.path, decoder.args(outPath, ivfPath)...).CombinedOutput()
		if err != nil {
			t.Fatalf("%s %s: %v\n%s", name, decoder.name, err, out)
		}
		got, err := os.ReadFile(outPath)
		if err != nil {
			t.Fatalf("%s read %s output: %v", name, decoder.name, err)
		}
		if !bytes.Equal(got, want) {
			offset := firstByteDiff(got, want)
			var gotByte, wantByte byte
			if offset >= 0 && offset < len(got) {
				gotByte = got[offset]
			}
			if offset >= 0 && offset < len(want) {
				wantByte = want[offset]
			}
			t.Fatalf("%s %s output len=%d want len=%d first_diff=%d got=%#02x want=%#02x",
				name, decoder.name, len(got), len(want), offset, gotByte, wantByte)
		}
		t.Logf("%s %s: %d frames bit-exact", name, decoder.name, frameCount)
	}
}

func appendFrameRawYUV(dst []byte, frame *goav1.Frame) []byte {
	if frame == nil {
		return dst
	}
	bytesPerSample := frame.Layout.BytesPerSample
	dst = appendFramePlaneRawYUV(dst, frame.Y, bytesPerSample)
	dst = appendFramePlaneRawYUV(dst, frame.U, bytesPerSample)
	dst = appendFramePlaneRawYUV(dst, frame.V, bytesPerSample)
	return dst
}

func appendFramePlaneRawYUV(dst []byte, plane goav1.FramePlane, bytesPerSample int) []byte {
	if plane.Width == 0 || plane.Height == 0 || len(plane.Pix) == 0 {
		return dst
	}
	rowBytes := plane.Width * bytesPerSample
	for row := 0; row < plane.Height; row++ {
		off := row * plane.Stride
		dst = append(dst, plane.Pix[off:off+rowBytes]...)
	}
	return dst
}

type referenceIVFFrame struct {
	timestamp uint64
	payload   []byte
}

func appendReferenceIVF(
	dst []byte, width int, height int, timebaseNum uint32, timebaseDen uint32, frames []referenceIVFFrame,
) []byte {
	dst = append(dst,
		'D', 'K', 'I', 'F',
		0, 0,
		32, 0,
		'A', 'V', '0', '1',
		byte(width), byte(width>>8),
		byte(height), byte(height>>8),
		byte(timebaseNum), byte(timebaseNum>>8), byte(timebaseNum>>16), byte(timebaseNum>>24),
		byte(timebaseDen), byte(timebaseDen>>8), byte(timebaseDen>>16), byte(timebaseDen>>24),
		byte(len(frames)), byte(len(frames)>>8), byte(len(frames)>>16), byte(len(frames)>>24),
		0, 0, 0, 0,
	)
	for _, frame := range frames {
		size := uint32(len(frame.payload))
		dst = append(dst,
			byte(size), byte(size>>8), byte(size>>16), byte(size>>24),
			byte(frame.timestamp), byte(frame.timestamp>>8), byte(frame.timestamp>>16), byte(frame.timestamp>>24),
			byte(frame.timestamp>>32), byte(frame.timestamp>>40), byte(frame.timestamp>>48), byte(frame.timestamp>>56),
		)
		dst = append(dst, frame.payload...)
	}
	return dst
}

func firstByteDiff(a []byte, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}
