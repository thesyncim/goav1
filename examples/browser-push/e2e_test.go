package main

import (
	"sync/atomic"
	"testing"
	"time"

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
	receiver, err := webrtc.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	defer receiver.Close()
	if _, err := receiver.AddTransceiverFromKind(webrtc.RTPCodecTypeVideo, webrtc.RTPTransceiverInit{
		Direction: webrtc.RTPTransceiverDirectionRecvonly,
	}); err != nil {
		t.Fatal(err)
	}

	type tu struct {
		data []byte
	}
	decoded := make(chan tu, 256)
	receiver.OnTrack(func(track *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		if track.Codec().MimeType != webrtc.MimeTypeAV1 {
			t.Errorf("negotiated %s, want AV1", track.Codec().MimeType)
			return
		}
		sb := samplebuilder.New(64, &codecs.AV1Depacketizer{}, track.Codec().ClockRate)
		for {
			pkt, _, err := track.ReadRTP()
			if err != nil {
				return
			}
			sb.Push(pkt)
			for s := sb.Pop(); s != nil; s = sb.Pop() {
				decoded <- tu{data: s.Data}
			}
		}
	})

	sender, err := webrtc.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	defer sender.Close()
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

	connected := make(chan struct{})
	sender.OnConnectionStateChange(func(s webrtc.PeerConnectionState) {
		if s == webrtc.PeerConnectionStateConnected {
			close(connected)
		}
	})
	select {
	case <-connected:
	case <-time.After(10 * time.Second):
		t.Fatal("peer connections never connected")
	}

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
	var tus [][]byte
	deadline := time.After(15 * time.Second)
	for len(tus) < 60 {
		select {
		case u := <-decoded:
			tus = append(tus, u.data)
		case <-deadline:
			t.Fatalf("only %d temporal units arrived", len(tus))
		}
	}
	// The sample builder needs a packet of context before it emits, so the
	// first reassembled units can start mid-sequence - the browser has the
	// same view and recovers at the next keyframe. Decode from the first
	// unit that carries a sequence header.
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
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode frame %d: %v", frames, err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
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
	avg := lumaSum / lumaCount
	if avg < 40 || avg > 220 {
		t.Fatalf("average luma %d looks like a blank picture", avg)
	}
	t.Logf("decoded %d frames, average sampled luma %d", frames, avg)
}
