package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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
		"browser-push",
		appendReferenceIVF(nil, width, height, fps, 1, ivfFrames),
		wantYUV,
		frames,
	)
	t.Logf("decoded %d frames, average sampled luma %d", frames, avg)
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
