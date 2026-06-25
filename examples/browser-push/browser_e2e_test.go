package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	cdpruntime "github.com/chromedp/cdproto/runtime"
	"github.com/chromedp/chromedp"
	"github.com/pion/rtcp"
	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4"

	goav1 "github.com/thesyncim/goav1"
)

const (
	browserE2EEnv        = "GOAV1_BROWSER_E2E"
	requireBrowserE2EEnv = "GOAV1_REQUIRE_WEBRTC_BROWSER"
	browserExecutableEnv = "GOAV1_BROWSER_EXECUTABLE"
)

func TestBrowserLiveAV1PlaybackStats(t *testing.T) {
	required := os.Getenv(requireBrowserE2EEnv) == "1"
	if !required && os.Getenv(browserE2EEnv) != "1" {
		t.Skipf("set %s=1 to run the live browser/libwebrtc AV1 playback gate", browserE2EEnv)
	}
	browserPath, err := browserExecutable()
	if err != nil {
		if required {
			t.Fatalf("required browser executable unavailable: %v", err)
		}
		t.Skip(err)
	}

	var mu sync.Mutex
	var peers []*webrtc.PeerConnection
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexHTML)
	})
	mux.HandleFunc("/offer", func(w http.ResponseWriter, r *http.Request) {
		err := handleOfferWithPeerConnectionHook(w, r, 4_000_000, func(pc *webrtc.PeerConnection) {
			mu.Lock()
			peers = append(peers, pc)
			mu.Unlock()
		})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	t.Cleanup(func() {
		mu.Lock()
		defer mu.Unlock()
		for _, pc := range peers {
			_ = pc.Close()
		}
	})

	ctx, cancel := context.WithTimeout(context.Background(), 35*time.Second)
	defer cancel()
	allocCtx, cancelAlloc := chromedp.NewExecAllocator(ctx,
		append(chromedp.DefaultExecAllocatorOptions[:],
			chromedp.ExecPath(browserPath),
			chromedp.Flag("headless", "new"),
			chromedp.Flag("autoplay-policy", "no-user-gesture-required"),
			chromedp.Flag("disable-background-timer-throttling", true),
			chromedp.Flag("disable-renderer-backgrounding", true),
			chromedp.Flag("mute-audio", true),
			chromedp.NoSandbox,
			chromedp.UserDataDir(t.TempDir()),
		)...,
	)
	defer cancelAlloc()
	browserCtx, cancelBrowser := chromedp.NewContext(allocCtx)
	defer cancelBrowser()

	for i, label := range []string{"initial", "reconnect"} {
		got := browserPlaybackEvidence{}
		if err := chromedp.Run(browserCtx,
			chromedp.Navigate(fmt.Sprintf("%s?session=%d", server.URL, i+1)),
			chromedp.Evaluate(browserPlaybackProbeJS(30), &got, evalAwaitPromise),
		); err != nil {
			t.Fatalf("%s browser AV1 playback probe: %v", label, err)
		}
		assertBrowserPlaybackEvidence(t, label, got)
	}
}

func TestBrowserLiveRTCEncoderDirectRTPPlaybackStats(t *testing.T) {
	required := os.Getenv(requireBrowserE2EEnv) == "1"
	if !required && os.Getenv(browserE2EEnv) != "1" {
		t.Skipf("set %s=1 to run the live browser/libwebrtc AV1 playback gate", browserE2EEnv)
	}
	browserPath, err := browserExecutable()
	if err != nil {
		if required {
			t.Fatalf("required browser executable unavailable: %v", err)
		}
		t.Skip(err)
	}

	scenarios := []struct {
		name         string
		query        string
		options      rtcEncoderRTPStreamOptions
		minKeyFrames int
	}{
		{
			name:         "default-l1t3",
			query:        "direct-rtp=1",
			options:      defaultRTCEncoderRTPStreamOptions(),
			minKeyFrames: 2,
		},
		{
			name:  "temporal-l1t1",
			query: "direct-rtp-l1t1=1",
			options: func() rtcEncoderRTPStreamOptions {
				options := defaultRTCEncoderRTPStreamOptions()
				options.ConfigForStep = rtcControlChurnConfigForScalabilityMode(goav1.EncoderScalabilityModeL1T1)
				return options
			}(),
			minKeyFrames: 2,
		},
		{
			name:  "temporal-l1t2",
			query: "direct-rtp-l1t2=1",
			options: func() rtcEncoderRTPStreamOptions {
				options := defaultRTCEncoderRTPStreamOptions()
				options.ConfigForStep = rtcControlChurnConfigForScalabilityMode(goav1.EncoderScalabilityModeL1T2)
				return options
			}(),
			minKeyFrames: 2,
		},
	}
	for _, scenario := range scenarios {
		scenario := scenario
		t.Run(scenario.name, func(t *testing.T) {
			got := runBrowserLiveRTCEncoderDirectRTPPlaybackStats(t, browserPath, scenario.name, scenario.query, scenario.options)
			if got.KeyFramesDecoded < scenario.minKeyFrames {
				t.Fatalf("%s browser keyframes=%d want at least %d after forced refresh",
					scenario.name, got.KeyFramesDecoded, scenario.minKeyFrames)
			}
		})
	}
}

func runBrowserLiveRTCEncoderDirectRTPPlaybackStats(
	t *testing.T, browserPath string, label string, query string, options rtcEncoderRTPStreamOptions,
) browserPlaybackEvidence {
	t.Helper()
	var mu sync.Mutex
	var peers []*webrtc.PeerConnection
	streamErr := make(chan error, 4)
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexHTML)
	})
	mux.HandleFunc("/offer", func(w http.ResponseWriter, r *http.Request) {
		err := handleRTCEncoderRTPOfferWithStreamOptions(w, r, func(pc *webrtc.PeerConnection) {
			mu.Lock()
			peers = append(peers, pc)
			mu.Unlock()
		}, streamErr, options, rtcSenderFeedbackOptions{})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	t.Cleanup(func() {
		mu.Lock()
		defer mu.Unlock()
		for _, pc := range peers {
			_ = pc.Close()
		}
	})

	ctx, cancel := context.WithTimeout(context.Background(), 35*time.Second)
	defer cancel()
	allocCtx, cancelAlloc := chromedp.NewExecAllocator(ctx,
		append(chromedp.DefaultExecAllocatorOptions[:],
			chromedp.ExecPath(browserPath),
			chromedp.Flag("headless", "new"),
			chromedp.Flag("autoplay-policy", "no-user-gesture-required"),
			chromedp.Flag("disable-background-timer-throttling", true),
			chromedp.Flag("disable-renderer-backgrounding", true),
			chromedp.Flag("mute-audio", true),
			chromedp.NoSandbox,
			chromedp.UserDataDir(t.TempDir()),
		)...,
	)
	defer cancelAlloc()
	browserCtx, cancelBrowser := chromedp.NewContext(allocCtx)
	defer cancelBrowser()

	got := browserPlaybackEvidence{}
	if err := chromedp.Run(browserCtx,
		chromedp.Navigate(server.URL+"?"+query),
		chromedp.Evaluate(browserPlaybackProbeJS(45), &got, evalAwaitPromise),
	); err != nil {
		t.Fatalf("%s browser AV1 playback probe: %v", label, err)
	}
	if !got.OK {
		select {
		case err := <-streamErr:
			t.Fatalf("%s stream failed before browser playback: %v; last=%+v", label, err, got)
		default:
		}
	}
	assertBrowserPlaybackEvidence(t, label, got)
	select {
	case err := <-streamErr:
		t.Fatalf("%s stream failed: %v", label, err)
	default:
	}
	return got
}

func TestBrowserLiveRTCEncoderDirectRTPImpairmentFeedback(t *testing.T) {
	required := os.Getenv(requireBrowserE2EEnv) == "1"
	if !required && os.Getenv(browserE2EEnv) != "1" {
		t.Skipf("set %s=1 to run the live browser/libwebrtc AV1 feedback gate", browserE2EEnv)
	}
	browserPath, err := browserExecutable()
	if err != nil {
		if required {
			t.Fatalf("required browser executable unavailable: %v", err)
		}
		t.Skip(err)
	}

	feedback := &rtcSenderFeedbackCounters{}
	var droppedPackets atomic.Int64
	var keyPictures atomic.Int64
	var postFeedbackKeyPictures atomic.Int64
	options := defaultRTCEncoderRTPStreamOptions()
	options.ForceKeyFrame = func(frameIndex int) bool { return false }
	options.DropPacket = func(frameIndex int, packetIndex int, _ rtp.Packet) bool {
		if frameIndex != 10 || packetIndex >= 4 {
			return false
		}
		droppedPackets.Add(1)
		return true
	}
	options.OnPicture = func(_ int, picture goav1.RTCPicture) {
		if !picture.Keyframe {
			return
		}
		keyPictures.Add(1)
		if rtcSenderFeedbackTotal(feedback) > 0 {
			postFeedbackKeyPictures.Add(1)
		}
	}

	var mu sync.Mutex
	var peers []*webrtc.PeerConnection
	streamErr := make(chan error, 4)
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexHTML)
	})
	mux.HandleFunc("/offer", func(w http.ResponseWriter, r *http.Request) {
		err := handleRTCEncoderRTPOfferWithStreamOptions(w, r, func(pc *webrtc.PeerConnection) {
			mu.Lock()
			peers = append(peers, pc)
			mu.Unlock()
		}, streamErr, options, rtcSenderFeedbackOptions{Counters: feedback})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	t.Cleanup(func() {
		mu.Lock()
		defer mu.Unlock()
		for _, pc := range peers {
			_ = pc.Close()
		}
	})

	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
	defer cancel()
	allocCtx, cancelAlloc := chromedp.NewExecAllocator(ctx,
		append(chromedp.DefaultExecAllocatorOptions[:],
			chromedp.ExecPath(browserPath),
			chromedp.Flag("headless", "new"),
			chromedp.Flag("autoplay-policy", "no-user-gesture-required"),
			chromedp.Flag("disable-background-timer-throttling", true),
			chromedp.Flag("disable-renderer-backgrounding", true),
			chromedp.Flag("mute-audio", true),
			chromedp.NoSandbox,
			chromedp.UserDataDir(t.TempDir()),
		)...,
	)
	defer cancelAlloc()
	browserCtx, cancelBrowser := chromedp.NewContext(allocCtx)
	defer cancelBrowser()

	got := browserPlaybackEvidence{}
	if err := chromedp.Run(browserCtx,
		chromedp.Navigate(server.URL+"?direct-rtp-loss=1"),
		chromedp.Evaluate(browserPlaybackProbeJS(45), &got, evalAwaitPromise),
	); err != nil {
		t.Fatalf("direct RTP impairment browser AV1 playback probe: %v", err)
	}
	if !got.OK {
		select {
		case err := <-streamErr:
			t.Fatalf("direct RTP impairment stream failed before browser playback: %v; last=%+v", err, got)
		default:
		}
	}
	assertBrowserPlaybackEvidence(t, "direct-rtp-impairment", got)
	if droppedPackets.Load() == 0 {
		t.Fatal("direct RTP impairment test did not drop any packets")
	}
	if rtcSenderFeedbackTotal(feedback) == 0 {
		t.Fatalf("direct RTP impairment produced no browser RTCP feedback after %d dropped packets", droppedPackets.Load())
	}
	if keyPictures.Load() < 2 || postFeedbackKeyPictures.Load() == 0 || got.KeyFramesDecoded < 2 {
		t.Fatalf("direct RTP impairment recovery keys server=%d postFeedback=%d browser=%d feedback=%s",
			keyPictures.Load(), postFeedbackKeyPictures.Load(), got.KeyFramesDecoded, rtcSenderFeedbackString(feedback))
	}
	select {
	case err := <-streamErr:
		t.Fatalf("direct RTP impairment stream failed: %v", err)
	default:
	}
	t.Logf("direct RTP impairment feedback: dropped=%d %s browserNACK=%d",
		droppedPackets.Load(), rtcSenderFeedbackString(feedback), got.NACKCount)
}

func TestBrowserLiveRTCEncoderDirectRTPNACKRetransmission(t *testing.T) {
	required := os.Getenv(requireBrowserE2EEnv) == "1"
	if !required && os.Getenv(browserE2EEnv) != "1" {
		t.Skipf("set %s=1 to run the live browser/libwebrtc AV1 retransmission gate", browserE2EEnv)
	}
	browserPath, err := browserExecutable()
	if err != nil {
		if required {
			t.Fatalf("required browser executable unavailable: %v", err)
		}
		t.Skip(err)
	}

	cacheSlots := make([]goav1.RTPRetransmissionCacheSlot, 4096)
	for i := range cacheSlots {
		cacheSlots[i].Packet = make([]byte, 0, 1600)
	}
	retransmitCache, err := goav1.BindRTPRetransmissionCache(cacheSlots)
	if err != nil {
		t.Fatalf("BindRTPRetransmissionCache: %v", err)
	}

	feedback := &rtcSenderFeedbackCounters{}
	var cacheMu sync.Mutex
	var droppedPackets atomic.Int64
	var repairedPackets atomic.Int64
	var repairMisses atomic.Int64
	var keyPictures atomic.Int64
	var postFeedbackKeyPictures atomic.Int64
	options := defaultRTCEncoderRTPStreamOptions()
	options.ForceKeyFrame = func(frameIndex int) bool { return false }
	options.DropPacket = func(frameIndex int, packetIndex int, _ rtp.Packet) bool {
		if frameIndex != 10 || packetIndex >= 4 {
			return false
		}
		droppedPackets.Add(1)
		return true
	}
	options.OnPacket = func(_ int, _ int, packet rtp.Packet, _ bool) error {
		raw := make([]byte, packet.MarshalSize())
		n, err := packet.MarshalTo(raw)
		if err != nil {
			return err
		}
		cacheMu.Lock()
		defer cacheMu.Unlock()
		return retransmitCache.Store(raw[:n])
	}
	options.OnPicture = func(_ int, picture goav1.RTCPicture) {
		if !picture.Keyframe {
			return
		}
		keyPictures.Add(1)
		if rtcSenderFeedbackTotal(feedback) > 0 {
			postFeedbackKeyPictures.Add(1)
		}
	}

	retransmitBuf := make([]byte, 0, 1600*64)
	retransmitSpans := make([]goav1.RTPRetransmissionPacketSpan, 64)
	nackSeqs := make([]uint16, 0, 64)
	repairNACK := func(track *webrtc.TrackLocalStaticRTP, nack *rtcp.TransportLayerNack) bool {
		if track == nil || len(nack.Nacks) == 0 {
			return false
		}
		pairs := make([]goav1.RTCPGenericNACKPair, len(nack.Nacks))
		for i := range nack.Nacks {
			pairs[i] = goav1.RTCPGenericNACKPair{
				PacketID:          nack.Nacks[i].PacketID,
				LostPacketBitmask: uint16(nack.Nacks[i].LostPackets),
			}
		}
		var err error
		nackSeqs, err = goav1.AppendRTCPGenericNACKPairSequenceNumbers(nackSeqs[:0], pairs)
		if err != nil {
			repairMisses.Add(1)
			return false
		}
		cacheMu.Lock()
		out, count, err := retransmitCache.AppendPacketsForRTCPGenericNACKPairs(
			retransmitBuf[:0], retransmitSpans, pairs)
		cacheMu.Unlock()
		if err != nil {
			repairMisses.Add(1)
			return false
		}
		for i := 0; i < count; i++ {
			span := retransmitSpans[i]
			if _, err := track.Write(out[span.Offset : span.Offset+span.Length]); err != nil {
				repairMisses.Add(1)
				return false
			}
			repairedPackets.Add(1)
		}
		if count < len(nackSeqs) {
			repairMisses.Add(int64(len(nackSeqs) - count))
			return false
		}
		return count > 0
	}

	var mu sync.Mutex
	var peers []*webrtc.PeerConnection
	streamErr := make(chan error, 4)
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexHTML)
	})
	mux.HandleFunc("/offer", func(w http.ResponseWriter, r *http.Request) {
		err := handleRTCEncoderRTPOfferWithStreamOptions(w, r, func(pc *webrtc.PeerConnection) {
			mu.Lock()
			peers = append(peers, pc)
			mu.Unlock()
		}, streamErr, options, rtcSenderFeedbackOptions{Counters: feedback, OnNACK: repairNACK})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	t.Cleanup(func() {
		mu.Lock()
		defer mu.Unlock()
		for _, pc := range peers {
			_ = pc.Close()
		}
	})

	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
	defer cancel()
	allocCtx, cancelAlloc := chromedp.NewExecAllocator(ctx,
		append(chromedp.DefaultExecAllocatorOptions[:],
			chromedp.ExecPath(browserPath),
			chromedp.Flag("headless", "new"),
			chromedp.Flag("autoplay-policy", "no-user-gesture-required"),
			chromedp.Flag("disable-background-timer-throttling", true),
			chromedp.Flag("disable-renderer-backgrounding", true),
			chromedp.Flag("mute-audio", true),
			chromedp.NoSandbox,
			chromedp.UserDataDir(t.TempDir()),
		)...,
	)
	defer cancelAlloc()
	browserCtx, cancelBrowser := chromedp.NewContext(allocCtx)
	defer cancelBrowser()

	got := browserPlaybackEvidence{}
	if err := chromedp.Run(browserCtx,
		chromedp.Navigate(server.URL+"?direct-rtp-retransmit=1"),
		chromedp.Evaluate(browserPlaybackProbeJS(45), &got, evalAwaitPromise),
	); err != nil {
		t.Fatalf("direct RTP retransmission browser AV1 playback probe: %v", err)
	}
	if !got.OK {
		select {
		case err := <-streamErr:
			t.Fatalf("direct RTP retransmission stream failed before browser playback: %v; last=%+v", err, got)
		default:
		}
	}
	assertBrowserPlaybackEvidence(t, "direct-rtp-retransmission", got)
	if droppedPackets.Load() == 0 {
		t.Fatal("direct RTP retransmission test did not drop any packets")
	}
	if feedback.NACK.Load() == 0 && got.NACKCount == 0 {
		t.Fatalf("direct RTP retransmission produced no browser NACK after %d dropped packets", droppedPackets.Load())
	}
	if repairedPackets.Load() == 0 {
		t.Fatalf("direct RTP retransmission did not resend cached packets after %d dropped packets", droppedPackets.Load())
	}
	if repairMisses.Load() != 0 {
		t.Fatalf("direct RTP retransmission repair misses=%d repaired=%d feedback=%s",
			repairMisses.Load(), repairedPackets.Load(), rtcSenderFeedbackString(feedback))
	}
	if postFeedbackKeyPictures.Load() != 0 || got.KeyFramesDecoded > 1 {
		t.Fatalf("direct RTP retransmission forced key recovery instead of packet repair: serverKeys=%d postFeedback=%d browserKeys=%d feedback=%s",
			keyPictures.Load(), postFeedbackKeyPictures.Load(), got.KeyFramesDecoded, rtcSenderFeedbackString(feedback))
	}
	select {
	case err := <-streamErr:
		t.Fatalf("direct RTP retransmission stream failed: %v", err)
	default:
	}
	t.Logf("direct RTP retransmission: dropped=%d repaired=%d %s browserNACK=%d",
		droppedPackets.Load(), repairedPackets.Load(), rtcSenderFeedbackString(feedback), got.NACKCount)
}

type browserPlaybackEvidence struct {
	OK                    bool   `json:"ok"`
	Error                 string `json:"error"`
	ConnectionState       string `json:"connectionState"`
	ICEConnectionState    string `json:"iceConnectionState"`
	PageError             string `json:"pageError"`
	VideoReadyState       int    `json:"videoReadyState"`
	VideoCurrentTimeMS    int    `json:"videoCurrentTimeMS"`
	VideoWidth            int    `json:"videoWidth"`
	VideoHeight           int    `json:"videoHeight"`
	FramesDecoded         int    `json:"framesDecoded"`
	KeyFramesDecoded      int    `json:"keyFramesDecoded"`
	FramesReceived        int    `json:"framesReceived"`
	PacketsReceived       int    `json:"packetsReceived"`
	BytesReceived         int    `json:"bytesReceived"`
	PLICount              int    `json:"pliCount"`
	FIRCount              int    `json:"firCount"`
	NACKCount             int    `json:"nackCount"`
	FreezeCount           int    `json:"freezeCount"`
	JitterMS              int    `json:"jitterMS"`
	CodecMimeType         string `json:"codecMimeType"`
	DecoderImplementation string `json:"decoderImplementation"`
}

func assertBrowserPlaybackEvidence(t *testing.T, label string, got browserPlaybackEvidence) {
	t.Helper()
	if !got.OK {
		t.Fatalf("%s browser AV1 playback probe failed: %s; last=%+v", label, got.Error, got)
	}
	if got.VideoWidth != width || got.VideoHeight != height {
		t.Fatalf("%s browser decoded size=%dx%d want %dx%d", label, got.VideoWidth, got.VideoHeight, width, height)
	}
	if got.FramesDecoded < 30 || got.PacketsReceived == 0 || got.BytesReceived == 0 {
		t.Fatalf("%s browser stats frames=%d packets=%d bytes=%d",
			label, got.FramesDecoded, got.PacketsReceived, got.BytesReceived)
	}
	if got.CodecMimeType != "" && got.CodecMimeType != "video/AV1" {
		t.Fatalf("%s browser codec mime=%q want video/AV1", label, got.CodecMimeType)
	}
	t.Logf("%s browser AV1 playback: frames=%d keyframes=%d packets=%d bytes=%d decoder=%q pli=%d fir=%d nack=%d",
		label, got.FramesDecoded, got.KeyFramesDecoded, got.PacketsReceived, got.BytesReceived,
		got.DecoderImplementation, got.PLICount, got.FIRCount, got.NACKCount)
}

func browserPlaybackProbeJS(minFrames int) string {
	return fmt.Sprintf(`(async () => {
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  let last = {};
  const snapshot = async () => {
    const pc = window.goav1PC;
    const video = window.goav1Video || document.getElementById('video');
    const out = {
      ok: false,
      error: '',
      connectionState: pc ? pc.connectionState : 'missing',
      iceConnectionState: pc ? pc.iceConnectionState : 'missing',
      pageError: window.goav1Error || '',
      videoReadyState: video ? video.readyState : 0,
      videoCurrentTimeMS: video ? Math.round(video.currentTime * 1000) : 0,
      videoWidth: video ? video.videoWidth : 0,
      videoHeight: video ? video.videoHeight : 0,
      framesDecoded: video && video.webkitDecodedFrameCount ? Number(video.webkitDecodedFrameCount) : 0,
      keyFramesDecoded: 0,
      framesReceived: 0,
      packetsReceived: 0,
      bytesReceived: 0,
      pliCount: 0,
      firCount: 0,
      nackCount: 0,
      freezeCount: 0,
      jitterMS: 0,
      codecMimeType: '',
      decoderImplementation: '',
    };
    if (!pc) return out;
    const stats = await pc.getStats();
    const codecs = new Map();
    stats.forEach((report) => {
      if (report.type === 'codec') codecs.set(report.id, report);
    });
    stats.forEach((report) => {
      if (report.type !== 'inbound-rtp' || (report.kind !== 'video' && report.mediaType !== 'video')) return;
      out.framesDecoded = Math.max(out.framesDecoded, Number(report.framesDecoded || 0));
      out.keyFramesDecoded = Number(report.keyFramesDecoded || 0);
      out.framesReceived = Number(report.framesReceived || 0);
      out.packetsReceived = Number(report.packetsReceived || 0);
      out.bytesReceived = Number(report.bytesReceived || 0);
      out.pliCount = Number(report.pliCount || 0);
      out.firCount = Number(report.firCount || 0);
      out.nackCount = Number(report.nackCount || 0);
      out.freezeCount = Number(report.freezeCount || 0);
      out.jitterMS = Math.round(Number(report.jitter || 0) * 1000);
      out.decoderImplementation = String(report.decoderImplementation || '');
      const codec = codecs.get(report.codecId);
      if (codec) out.codecMimeType = String(codec.mimeType || '');
    });
    return out;
  };
  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    last = await snapshot();
    if (last.pageError) return Object.assign(last, { error: last.pageError });
    if (
      last.connectionState === 'connected' &&
      last.videoReadyState >= 2 &&
      last.videoCurrentTimeMS >= 250 &&
      last.videoWidth > 0 &&
      last.videoHeight > 0 &&
      last.framesDecoded >= %d &&
      last.packetsReceived > 0 &&
      last.bytesReceived > 0
    ) {
      return Object.assign(last, { ok: true });
    }
    await sleep(250);
  }
  return Object.assign(last, { error: 'timed out waiting for live AV1 frames decoded by browser' });
})()`, minFrames)
}

func evalAwaitPromise(p *cdpruntime.EvaluateParams) *cdpruntime.EvaluateParams {
	return p.WithAwaitPromise(true)
}

func handleRTCEncoderRTPOfferWithPeerConnectionHook(
	w http.ResponseWriter,
	r *http.Request,
	onPeerConnection func(*webrtc.PeerConnection),
	streamErr chan<- error,
) error {
	return handleRTCEncoderRTPOfferWithStreamOptions(
		w, r, onPeerConnection, streamErr, defaultRTCEncoderRTPStreamOptions(), rtcSenderFeedbackOptions{})
}

func handleRTCEncoderRTPOfferWithStreamOptions(
	w http.ResponseWriter,
	r *http.Request,
	onPeerConnection func(*webrtc.PeerConnection),
	streamErr chan<- error,
	streamOptions rtcEncoderRTPStreamOptions,
	feedback rtcSenderFeedbackOptions,
) error {
	offerBytes, err := io.ReadAll(r.Body)
	if err != nil {
		return err
	}
	pc, err := newRTCEncoderRTPPeerConnection()
	if err != nil {
		return err
	}
	if onPeerConnection != nil {
		onPeerConnection(pc)
	}
	track, err := webrtc.NewTrackLocalStaticRTP(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeAV1}, "video", "goav1")
	if err != nil {
		pc.Close()
		return err
	}
	sender, err := pc.AddTrack(track)
	if err != nil {
		pc.Close()
		return err
	}
	var wantKey atomic.Bool
	wantKey.Store(true)
	done := make(chan struct{})
	startSenderFeedbackReaderWithOptions(sender, track, &wantKey, done, feedback)
	pc.OnConnectionStateChange(func(s webrtc.PeerConnectionState) {
		switch s {
		case webrtc.PeerConnectionStateConnected:
			extensions, err := rtcRTPHeaderExtensionsFromAnswer(pc.LocalDescription().SDP)
			if err != nil {
				select {
				case streamErr <- err:
				default:
				}
				return
			}
			options := streamOptions
			options.HeaderExtensions = extensions
			go func() {
				if err := streamRTCEncoderRTPWithOptions(track, &wantKey, done, options); err != nil {
					select {
					case streamErr <- err:
					default:
					}
				}
			}()
		case webrtc.PeerConnectionStateFailed, webrtc.PeerConnectionStateClosed,
			webrtc.PeerConnectionStateDisconnected:
			select {
			case <-done:
			default:
				close(done)
			}
		}
	})
	if err := pc.SetRemoteDescription(webrtc.SessionDescription{
		Type: webrtc.SDPTypeOffer, SDP: string(offerBytes),
	}); err != nil {
		pc.Close()
		return err
	}
	answer, err := pc.CreateAnswer(nil)
	if err != nil {
		pc.Close()
		return err
	}
	gathered := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(answer); err != nil {
		pc.Close()
		return err
	}
	<-gathered
	fmt.Fprint(w, pc.LocalDescription().SDP)
	return nil
}

func rtcSenderFeedbackTotal(counters *rtcSenderFeedbackCounters) int64 {
	if counters == nil {
		return 0
	}
	return counters.PictureLoss.Load() + counters.FullIntra.Load() + counters.NACK.Load()
}

func rtcSenderFeedbackString(counters *rtcSenderFeedbackCounters) string {
	if counters == nil {
		return "pli=0 fir=0 nack=0"
	}
	return fmt.Sprintf("pli=%d fir=%d nack=%d",
		counters.PictureLoss.Load(), counters.FullIntra.Load(), counters.NACK.Load())
}

func newRTCEncoderRTPPeerConnection() (*webrtc.PeerConnection, error) {
	var mediaEngine webrtc.MediaEngine
	if err := mediaEngine.RegisterDefaultCodecs(); err != nil {
		return nil, err
	}
	if err := mediaEngine.RegisterHeaderExtension(
		webrtc.RTPHeaderExtensionCapability{URI: goav1.AV1RTPDependencyDescriptorURI},
		webrtc.RTPCodecTypeVideo,
	); err != nil {
		return nil, err
	}
	api := webrtc.NewAPI(webrtc.WithMediaEngine(&mediaEngine))
	return api.NewPeerConnection(webrtc.Configuration{})
}

func rtcRTPHeaderExtensionsFromAnswer(answerSDP string) (rtcRTPHeaderExtensions, error) {
	extensions := rtcRTPHeaderExtensions{
		Profile: goav1.RTPExtensionProfileTwoByte,
	}
	if dependencyID, ok := goav1.AV1SDPAnswersSendHeaderExtensionID(
		answerSDP, goav1.AV1RTPDependencyDescriptorURI); ok {
		if dependencyID <= 0 || dependencyID > 255 {
			return rtcRTPHeaderExtensions{}, fmt.Errorf("answer dependency descriptor extmap id=%d", dependencyID)
		}
		extensions.DependencyDescriptorID = uint8(dependencyID)
	}
	if transportCCID, ok := goav1.AV1SDPAnswersSendHeaderExtensionID(
		answerSDP, goav1.AV1RTPTransportWideCCURI); ok {
		if transportCCID <= 0 || transportCCID > 255 {
			return rtcRTPHeaderExtensions{}, fmt.Errorf("answer transport-wide-cc extmap id=%d", transportCCID)
		}
		extensions.TransportWideCCID = uint8(transportCCID)
	} else if transportCC02ID, ok := goav1.AV1SDPAnswersSendHeaderExtensionID(
		answerSDP, goav1.AV1RTPTransportWideCC02URI); ok {
		if transportCC02ID <= 0 || transportCC02ID > 255 {
			return rtcRTPHeaderExtensions{}, fmt.Errorf("answer transport-wide-cc-02 extmap id=%d", transportCC02ID)
		}
		extensions.TransportWideCC02ID = uint8(transportCC02ID)
	}
	return extensions, nil
}

func browserExecutable() (string, error) {
	if p := os.Getenv(browserExecutableEnv); p != "" {
		if _, err := os.Stat(p); err != nil {
			return "", fmt.Errorf("%s=%q: %w", browserExecutableEnv, p, err)
		}
		return p, nil
	}
	candidates := []string{"google-chrome", "chromium", "chromium-browser", "chrome"}
	switch runtime.GOOS {
	case "darwin":
		candidates = append([]string{
			"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
			"/Applications/Chromium.app/Contents/MacOS/Chromium",
			"/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
		}, candidates...)
	case "linux":
		candidates = append([]string{
			"/usr/bin/google-chrome",
			"/usr/bin/google-chrome-stable",
			"/usr/bin/chromium",
			"/usr/bin/chromium-browser",
		}, candidates...)
	case "windows":
		candidates = append([]string{
			`C:\Program Files\Google\Chrome\Application\chrome.exe`,
			`C:\Program Files (x86)\Google\Chrome\Application\chrome.exe`,
			`C:\Program Files\Microsoft\Edge\Application\msedge.exe`,
		}, candidates...)
	}
	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
		if path, err := exec.LookPath(candidate); err == nil {
			return path, nil
		}
	}
	return "", errors.New("Chrome/Chromium executable not found; set GOAV1_BROWSER_EXECUTABLE")
}
