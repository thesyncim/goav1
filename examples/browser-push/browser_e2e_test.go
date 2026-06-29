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
	"sort"
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
	browserCtx, closeBrowser := newBrowserE2EContext(t, ctx, browserPath)
	defer closeBrowser()

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

	for _, scenario := range browserRTCEncoderDirectRTPPlaybackScenarios(t) {
		scenario := scenario
		t.Run(scenario.name, func(t *testing.T) {
			got := runBrowserLiveRTCEncoderDirectRTPPlaybackStats(
				t, browserPath, scenario.name, scenario.query, scenario.options, scenario.wantWidth, scenario.wantHeight)
			if got.KeyFramesDecoded < scenario.minKeyFrames {
				t.Fatalf("%s browser keyframes=%d want at least %d after forced refresh",
					scenario.name, got.KeyFramesDecoded, scenario.minKeyFrames)
			}
		})
	}
}

func TestBrowserLiveRTCEncoderDirectRTPRepeatedPlaybackSoak(t *testing.T) {
	required := os.Getenv(requireBrowserE2EEnv) == "1"
	if !required && os.Getenv(browserE2EEnv) != "1" {
		t.Skipf("set %s=1 to run the repeated live browser/libwebrtc AV1 playback gate", browserE2EEnv)
	}
	browserPath, err := browserExecutable()
	if err != nil {
		if required {
			t.Fatalf("required browser executable unavailable: %v", err)
		}
		t.Skip(err)
	}

	allScenarios := browserRTCEncoderDirectRTPPlaybackScenarios(t)
	soakScenarios := []browserRTCEncoderDirectRTPPlaybackScenario{
		browserRTCEncoderDirectRTPPlaybackScenarioByName(t, allScenarios, "direct-L1T3"),
		browserRTCEncoderDirectRTPPlaybackScenarioByName(t, allScenarios, "shared-svc-forward-base-L3T3_KEY_SHIFT"),
		browserRTCEncoderDirectRTPPlaybackScenarioByName(t, allScenarios, "simulcast-forward-top-S3T3h"),
	}

	for _, scenario := range soakScenarios {
		scenario := scenario
		t.Run(scenario.name, func(t *testing.T) {
			for loop := 0; loop < 2; loop++ {
				loop := loop
				t.Run(fmt.Sprintf("loop-%d", loop+1), func(t *testing.T) {
					query := fmt.Sprintf("%s&soak-loop=%d", scenario.query, loop+1)
					label := fmt.Sprintf("%s-loop-%d", scenario.name, loop+1)
					got := runBrowserLiveRTCEncoderDirectRTPPlaybackStats(
						t, browserPath, label, query, scenario.options, scenario.wantWidth, scenario.wantHeight)
					if got.KeyFramesDecoded < scenario.minKeyFrames {
						t.Fatalf("%s browser keyframes=%d want at least %d after repeated playback",
							label, got.KeyFramesDecoded, scenario.minKeyFrames)
					}
				})
			}
		})
	}
}

func TestBrowserLiveRTCEncoderDirectRTPControlChurnPlayback(t *testing.T) {
	required := os.Getenv(requireBrowserE2EEnv) == "1"
	if !required && os.Getenv(browserE2EEnv) != "1" {
		t.Skipf("set %s=1 to run the live browser/libwebrtc AV1 control-churn gate", browserE2EEnv)
	}
	browserPath, err := browserExecutable()
	if err != nil {
		if required {
			t.Fatalf("required browser executable unavailable: %v", err)
		}
		t.Skip(err)
	}

	allScenarios := browserRTCEncoderDirectRTPPlaybackScenarios(t)
	scenarios := []browserRTCEncoderDirectRTPControlChurnScenario{
		{
			name:       "direct-l1-scalability-controls",
			query:      "control-churn=direct-l1-scalability",
			options:    browserRTCEncoderDirectRTPControlChurnOptions(browserRTCEncoderDirectRTPControlChurnSingleSpatialConfig),
			wantWidth:  width,
			wantHeight: height,
			wantModes: []goav1.EncoderScalabilityMode{
				goav1.EncoderScalabilityModeL1T1,
				goav1.EncoderScalabilityModeL1T2,
				goav1.EncoderScalabilityModeL1T3,
			},
		},
		browserRTCEncoderDirectRTPControlChurnScenarioFromPlayback(
			browserRTCEncoderDirectRTPPlaybackScenarioByName(t, allScenarios, "shared-svc-forward-base-L3T3_KEY_SHIFT"),
			"shared-svc-base-controls"),
		browserRTCEncoderDirectRTPControlChurnScenarioFromPlayback(
			browserRTCEncoderDirectRTPPlaybackScenarioByName(t, allScenarios, "simulcast-forward-top-S3T3h"),
			"simulcast-top-controls"),
	}

	for _, scenario := range scenarios {
		scenario := scenario
		t.Run(scenario.name, func(t *testing.T) {
			evidence := &browserRTCEncoderDirectRTPControlChurnEvidence{}
			options := scenario.options
			evidence.attach(&options)
			got := runBrowserLiveRTCEncoderDirectRTPPlaybackStats(
				t, browserPath, scenario.name, scenario.query, options, scenario.wantWidth, scenario.wantHeight)
			if got.KeyFramesDecoded < 2 {
				t.Fatalf("%s browser keyframes=%d want at least 2 during control churn",
					scenario.name, got.KeyFramesDecoded)
			}
			evidence.assert(t, scenario.name, scenario.wantModes)
		})
	}
}

func TestBrowserLiveRTCEncoderDirectRTPReceiverEstimatedMaximumBitrateControl(t *testing.T) {
	required := os.Getenv(requireBrowserE2EEnv) == "1"
	if !required && os.Getenv(browserE2EEnv) != "1" {
		t.Skipf("set %s=1 to run the live browser/libwebrtc AV1 REMB control gate", browserE2EEnv)
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
	var rembApplied atomic.Int64
	var lastREMBTargetKbps atomic.Int64
	options := defaultRTCEncoderRTPStreamOptions()
	options.DisableTransportWideCC = true
	options.ForceKeyFrame = func(frameIndex int) bool { return false }
	options.DropPacket = func(frameIndex int, packetIndex int, _ rtp.Packet) bool {
		if frameIndex != 10 || packetIndex >= 4 {
			return false
		}
		droppedPackets.Add(1)
		return true
	}
	options.OnConfigApplied = func(_ int, step int, cfg goav1.EncoderConfig) {
		if step >= 0 {
			return
		}
		rembApplied.Add(1)
		lastREMBTargetKbps.Store(int64(cfg.TargetBitrateKbps))
	}

	got := runBrowserLiveRTCEncoderDirectRTPPlaybackStatsWithFeedback(
		t,
		browserPath,
		"direct-rtp-remb",
		"direct-rtp-remb=1",
		options,
		width,
		height,
		rtcSenderFeedbackOptions{Counters: feedback},
	)
	if got.KeyFramesDecoded < 1 {
		t.Fatalf("direct RTP REMB browser keyframes=%d want at least 1", got.KeyFramesDecoded)
	}
	if droppedPackets.Load() == 0 {
		t.Fatal("direct RTP REMB test did not drop any packets")
	}
	if feedback.ReceiverEstimatedMaximumBitrate.Load() == 0 {
		t.Fatalf("direct RTP REMB test produced no browser REMB feedback after %d dropped packets; feedback=%s",
			droppedPackets.Load(), rtcSenderFeedbackString(feedback))
	}
	if rembApplied.Load() == 0 || lastREMBTargetKbps.Load() == 0 {
		t.Fatalf("direct RTP REMB feedback was not applied through RTCEncoder.SetConfig; feedback=%s applied=%d target=%d",
			rtcSenderFeedbackString(feedback), rembApplied.Load(), lastREMBTargetKbps.Load())
	}
	t.Logf("direct RTP REMB control: dropped=%d applied=%d targetKbps=%d %s",
		droppedPackets.Load(), rembApplied.Load(), lastREMBTargetKbps.Load(), rtcSenderFeedbackString(feedback))
}

func TestBrowserLiveRTCEncoderDirectRTPTransportWideCCFeedback(t *testing.T) {
	required := os.Getenv(requireBrowserE2EEnv) == "1"
	if !required && os.Getenv(browserE2EEnv) != "1" {
		t.Skipf("set %s=1 to run the live browser/libwebrtc AV1 transport-cc gate", browserE2EEnv)
	}
	browserPath, err := browserExecutable()
	if err != nil {
		if required {
			t.Fatalf("required browser executable unavailable: %v", err)
		}
		t.Skip(err)
	}

	feedback := &rtcSenderFeedbackCounters{}
	var negotiatedTWCCID atomic.Int32
	var sentPackets atomic.Int64
	var sentTWCCPackets atomic.Int64
	var reportedPacketStatuses atomic.Int64
	var reportedRecvDeltas atomic.Int64
	var lastBaseSequence atomic.Uint32
	options := defaultRTCEncoderRTPStreamOptions()
	options.OnHeaderExtensions = func(extensions rtcRTPHeaderExtensions) {
		switch {
		case extensions.TransportWideCCID != 0:
			negotiatedTWCCID.Store(int32(extensions.TransportWideCCID))
		case extensions.TransportWideCC02ID != 0:
			negotiatedTWCCID.Store(int32(extensions.TransportWideCC02ID))
		}
	}
	options.OnPacket = func(_ int, _ int, packet rtp.Packet, dropped bool) error {
		if dropped {
			return nil
		}
		sentPackets.Add(1)
		if id := negotiatedTWCCID.Load(); id != 0 && len(packet.GetExtension(uint8(id))) > 0 {
			sentTWCCPackets.Add(1)
		}
		return nil
	}

	got := runBrowserLiveRTCEncoderDirectRTPPlaybackStatsWithFeedbackFrames(
		t,
		browserPath,
		"direct-rtp-twcc",
		"direct-rtp-twcc=1",
		options,
		width,
		height,
		rtcSenderFeedbackOptions{
			Counters: feedback,
			OnTransportLayerCC: func(tcc *rtcp.TransportLayerCC) {
				reportedPacketStatuses.Add(int64(tcc.PacketStatusCount))
				reportedRecvDeltas.Add(int64(len(tcc.RecvDeltas)))
				lastBaseSequence.Store(uint32(tcc.BaseSequenceNumber))
			},
		},
		90,
	)
	if got.KeyFramesDecoded < 1 {
		t.Fatalf("direct RTP transport-cc browser keyframes=%d want at least 1", got.KeyFramesDecoded)
	}
	if negotiatedTWCCID.Load() == 0 {
		t.Fatal("direct RTP transport-cc was not negotiated in the browser answer")
	}
	if sentPackets.Load() == 0 || sentTWCCPackets.Load() == 0 {
		t.Fatalf("direct RTP transport-cc packets sent=%d twccExtensionPackets=%d negotiatedID=%d",
			sentPackets.Load(), sentTWCCPackets.Load(), negotiatedTWCCID.Load())
	}
	if feedback.TransportLayerCC.Load() == 0 || reportedPacketStatuses.Load() == 0 {
		t.Fatalf("direct RTP transport-cc produced no browser TransportLayerCC feedback; sentTWCC=%d feedback=%s statuses=%d",
			sentTWCCPackets.Load(), rtcSenderFeedbackString(feedback), reportedPacketStatuses.Load())
	}
	t.Logf("direct RTP transport-cc: sent=%d twccPackets=%d reports=%d statuses=%d deltas=%d base=%d %s",
		sentPackets.Load(), sentTWCCPackets.Load(), feedback.TransportLayerCC.Load(),
		reportedPacketStatuses.Load(), reportedRecvDeltas.Load(), lastBaseSequence.Load(),
		rtcSenderFeedbackString(feedback))
}

func TestBrowserRTCEncoderDirectRTPPlaybackScenarios(t *testing.T) {
	scenarios := browserRTCEncoderDirectRTPPlaybackScenarios(t)
	seen := make(map[string]bool, len(scenarios))
	for _, scenario := range scenarios {
		if seen[scenario.name] {
			t.Fatalf("duplicate browser direct-RTP scenario %q", scenario.name)
		}
		seen[scenario.name] = true
	}

	for _, name := range []string{
		"direct-L1T3",
		"shared-svc-forward-base-L2T3_KEY_SHIFT",
		"shared-svc-forward-base-L3T2_KEY_SHIFT",
		"shared-svc-forward-base-L3T3_KEY_SHIFT",
		"simulcast-forward-top-S3T3h",
	} {
		if !seen[name] {
			t.Fatalf("browser direct-RTP scenario %q not found", name)
		}
	}
}

type browserRTCEncoderDirectRTPPlaybackScenario struct {
	name         string
	query        string
	options      rtcEncoderRTPStreamOptions
	wantWidth    int
	wantHeight   int
	minKeyFrames int
}

var browserRTCEncoderDirectRTPLocalKeyShiftModes = [...]goav1.EncoderScalabilityMode{
	goav1.EncoderScalabilityModeL2T3_KEY_SHIFT,
	goav1.EncoderScalabilityModeL3T2_KEY_SHIFT,
	goav1.EncoderScalabilityModeL3T3_KEY_SHIFT,
}

type browserRTCEncoderDirectRTPControlChurnScenario struct {
	name       string
	query      string
	options    rtcEncoderRTPStreamOptions
	wantWidth  int
	wantHeight int
	wantModes  []goav1.EncoderScalabilityMode
}

type browserRTCEncoderDirectRTPControlChurnEvidence struct {
	mu              sync.Mutex
	configs         []goav1.EncoderConfig
	pictures        int
	keyPictures     int
	frameTimestamps map[int]uint32
}

func (e *browserRTCEncoderDirectRTPControlChurnEvidence) attach(options *rtcEncoderRTPStreamOptions) {
	prevConfig := options.OnConfigApplied
	options.OnConfigApplied = func(frameIndex int, step int, cfg goav1.EncoderConfig) {
		if prevConfig != nil {
			prevConfig(frameIndex, step, cfg)
		}
		e.mu.Lock()
		e.configs = append(e.configs, cfg)
		e.mu.Unlock()
	}
	prevPicture := options.OnPicture
	options.OnPicture = func(frameIndex int, picture goav1.RTCPicture) {
		if prevPicture != nil {
			prevPicture(frameIndex, picture)
		}
		e.mu.Lock()
		e.pictures++
		if picture.Keyframe {
			e.keyPictures++
		}
		e.mu.Unlock()
	}
	prevPacket := options.OnPacket
	options.OnPacket = func(frameIndex int, packetIndex int, packet rtp.Packet, dropped bool) error {
		if prevPacket != nil {
			if err := prevPacket(frameIndex, packetIndex, packet, dropped); err != nil {
				return err
			}
		}
		if dropped {
			return nil
		}
		e.mu.Lock()
		if e.frameTimestamps == nil {
			e.frameTimestamps = make(map[int]uint32)
		}
		if _, ok := e.frameTimestamps[frameIndex]; !ok {
			e.frameTimestamps[frameIndex] = packet.Timestamp
		}
		e.mu.Unlock()
		return nil
	}
}

func (e *browserRTCEncoderDirectRTPControlChurnEvidence) assert(t *testing.T, label string, wantModes []goav1.EncoderScalabilityMode) {
	t.Helper()
	e.mu.Lock()
	configs := append([]goav1.EncoderConfig(nil), e.configs...)
	pictures := e.pictures
	keyPictures := e.keyPictures
	frameTimestamps := make(map[int]uint32, len(e.frameTimestamps))
	for frameIndex, timestamp := range e.frameTimestamps {
		frameTimestamps[frameIndex] = timestamp
	}
	e.mu.Unlock()

	if len(configs) < 4 {
		t.Fatalf("%s applied configs=%d want at least 4", label, len(configs))
	}
	if pictures < 42 {
		t.Fatalf("%s encoded pictures=%d want at least 42", label, pictures)
	}
	if keyPictures < 2 {
		t.Fatalf("%s key pictures=%d want at least 2", label, keyPictures)
	}

	fps := make(map[goav1.EncoderRational]bool)
	targets := make(map[int32]bool)
	rateControls := make(map[goav1.EncoderRateControlMode]bool)
	contents := make(map[goav1.EncoderContentHint]bool)
	modes := make(map[goav1.EncoderScalabilityMode]bool)
	for _, cfg := range configs {
		fps[cfg.MaxFramerate] = true
		targets[cfg.TargetBitrateKbps] = true
		rateControls[cfg.RateControl] = true
		contents[cfg.Content] = true
		modes[cfg.Scalability] = true
	}
	if len(fps) < 3 {
		t.Fatalf("%s framerate values=%d configs=%v", label, len(fps), configs)
	}
	if len(targets) < 3 {
		t.Fatalf("%s target bitrate values=%d configs=%v", label, len(targets), configs)
	}
	if !rateControls[goav1.EncoderRateControlCBR] || !rateControls[goav1.EncoderRateControlCQP] {
		t.Fatalf("%s rate controls=%v want CBR and CQP", label, rateControls)
	}
	if !contents[goav1.EncoderContentCamera] || !contents[goav1.EncoderContentScreen] {
		t.Fatalf("%s content hints=%v want camera and screen", label, contents)
	}
	for _, mode := range wantModes {
		if !modes[mode] {
			t.Fatalf("%s missing scalability mode %s in applied configs %v", label, mode, configs)
		}
	}
	timestampDeltas := browserRTCEncoderDirectRTPFrameTimestampDeltas(frameTimestamps)
	if len(frameTimestamps) < 42 {
		t.Fatalf("%s RTP frame timestamps=%d want at least 42", label, len(frameTimestamps))
	}
	if len(timestampDeltas) < 3 {
		t.Fatalf("%s distinct RTP frame timestamp deltas=%d want at least 3", label, len(timestampDeltas))
	}
	matchedFPS := browserRTCEncoderDirectRTPMatchedFrameDurations(t, configs, timestampDeltas)
	if matchedFPS < 3 {
		t.Fatalf("%s RTP frame timestamp deltas matched %d configured framerates, want at least 3; deltas=%v configs=%v",
			label, matchedFPS, timestampDeltas, configs)
	}
	t.Logf("%s applied configs=%d pictures=%d keyPictures=%d fps=%d bitrates=%d rateControls=%d contents=%d modes=%d rtpFrames=%d rtpDeltas=%d matchedFPS=%d",
		label, len(configs), pictures, keyPictures, len(fps), len(targets), len(rateControls), len(contents), len(modes),
		len(frameTimestamps), len(timestampDeltas), matchedFPS)
}

func browserRTCEncoderDirectRTPFrameTimestampDeltas(frameTimestamps map[int]uint32) map[uint32]bool {
	frameIndexes := make([]int, 0, len(frameTimestamps))
	for frameIndex := range frameTimestamps {
		frameIndexes = append(frameIndexes, frameIndex)
	}
	sort.Ints(frameIndexes)
	deltas := make(map[uint32]bool)
	for i := 1; i < len(frameIndexes); i++ {
		prev, curr := frameIndexes[i-1], frameIndexes[i]
		if curr != prev+1 {
			continue
		}
		delta := frameTimestamps[curr] - frameTimestamps[prev]
		if delta != 0 {
			deltas[delta] = true
		}
	}
	return deltas
}

func browserRTCEncoderDirectRTPMatchedFrameDurations(
	t *testing.T, configs []goav1.EncoderConfig, timestampDeltas map[uint32]bool,
) int {
	t.Helper()
	matched := make(map[goav1.EncoderRational]bool)
	for _, cfg := range configs {
		if matched[cfg.MaxFramerate] {
			continue
		}
		duration, err := goav1.EncoderWebRTCRTPFrameDuration(cfg)
		if err != nil {
			t.Fatalf("EncoderWebRTCRTPFrameDuration(%+v): %v", cfg, err)
		}
		var remainder int64
		for i := 0; i < 4; i++ {
			if timestampDeltas[rtcTimestampIncrement(duration, &remainder)] {
				matched[cfg.MaxFramerate] = true
				break
			}
		}
	}
	return len(matched)
}

func browserRTCEncoderDirectRTPControlChurnOptions(configForStep func(step int) goav1.EncoderConfig) rtcEncoderRTPStreamOptions {
	options := defaultRTCEncoderRTPStreamOptions()
	options.ConfigForStep = configForStep
	return options
}

func browserRTCEncoderDirectRTPControlChurnSingleSpatialConfig(step int) goav1.EncoderConfig {
	cfg := rtcControlChurnConfig(step)
	modes := [...]goav1.EncoderScalabilityMode{
		goav1.EncoderScalabilityModeL1T1,
		goav1.EncoderScalabilityModeL1T2,
		goav1.EncoderScalabilityModeL1T3,
		goav1.EncoderScalabilityModeL1T2,
	}
	cfg.Scalability = modes[step%len(modes)]
	return cfg
}

func browserRTCEncoderDirectRTPControlChurnScenarioFromPlayback(
	scenario browserRTCEncoderDirectRTPPlaybackScenario, name string,
) browserRTCEncoderDirectRTPControlChurnScenario {
	return browserRTCEncoderDirectRTPControlChurnScenario{
		name:       name,
		query:      fmt.Sprintf("%s&control-churn=%s", scenario.query, name),
		options:    scenario.options,
		wantWidth:  scenario.wantWidth,
		wantHeight: scenario.wantHeight,
		wantModes:  []goav1.EncoderScalabilityMode{scenario.options.ConfigForStep(0).Scalability},
	}
}

func browserRTCEncoderDirectRTPPlaybackScenarios(t *testing.T) []browserRTCEncoderDirectRTPPlaybackScenario {
	t.Helper()
	modes := goav1.EncoderWebRTCScalabilityModes()
	scenarios := make([]browserRTCEncoderDirectRTPPlaybackScenario, 0, len(modes)+len(browserRTCEncoderDirectRTPLocalKeyShiftModes))
	for _, mode := range modes {
		scenarios = appendBrowserRTCEncoderDirectRTPPlaybackScenario(t, scenarios, mode)
	}
	for _, mode := range browserRTCEncoderDirectRTPLocalKeyShiftModes {
		scenarios = appendBrowserRTCEncoderDirectRTPPlaybackScenario(t, scenarios, mode)
	}
	return scenarios
}

func appendBrowserRTCEncoderDirectRTPPlaybackScenario(
	t *testing.T,
	scenarios []browserRTCEncoderDirectRTPPlaybackScenario,
	mode goav1.EncoderScalabilityMode,
) []browserRTCEncoderDirectRTPPlaybackScenario {
	t.Helper()
	spatialLayers, temporalLayers, _, ok := mode.Layers()
	if !ok {
		t.Fatalf("invalid WebRTC scalability mode %s", mode)
	}
	options := defaultRTCEncoderRTPStreamOptions()
	options.ConfigForStep = rtcControlChurnConfigForScalabilityMode(mode)
	cfg := options.ConfigForStep(0)
	normalized, err := goav1.SetWebRTCEncoderSVCConfig(cfg, 0, 0)
	if err != nil {
		t.Fatalf("normalize %s browser scenario: %v", mode, err)
	}

	delivery := "direct"
	targetSpatialID := uint8(0)
	if spatialLayers > 1 {
		if mode.IsSimulcast() {
			delivery = "simulcast-forward-top"
			targetSpatialID = spatialLayers - 1
		} else {
			delivery = "shared-svc-forward-base"
		}
		options.RTPOptionsForPicture = rtcActiveDecodeTargetOptionsForSpatialLayer(targetSpatialID, temporalLayers)
		options.FrameFilter = func(frame goav1.RTCFrame) bool { return frame.SpatialID == targetSpatialID }
	}
	layer := normalized.SpatialLayers[targetSpatialID]
	return append(scenarios, browserRTCEncoderDirectRTPPlaybackScenario{
		name:         fmt.Sprintf("%s-%s", delivery, mode),
		query:        fmt.Sprintf("direct-rtp-mode=%s", mode),
		options:      options,
		wantWidth:    browserAV1CodedDimension(int(layer.Resolution.Width)),
		wantHeight:   browserAV1CodedDimension(int(layer.Resolution.Height)),
		minKeyFrames: 2,
	})
}

func browserRTCEncoderDirectRTPPlaybackScenarioByName(t *testing.T, scenarios []browserRTCEncoderDirectRTPPlaybackScenario, name string) browserRTCEncoderDirectRTPPlaybackScenario {
	t.Helper()
	for _, scenario := range scenarios {
		if scenario.name == name {
			return scenario
		}
	}
	t.Fatalf("browser direct-RTP playback scenario %q not found", name)
	return browserRTCEncoderDirectRTPPlaybackScenario{}
}

func browserAV1CodedDimension(renderDimension int) int {
	return (renderDimension + 7) &^ 7
}

func runBrowserLiveRTCEncoderDirectRTPPlaybackStats(
	t *testing.T, browserPath string, label string, query string, options rtcEncoderRTPStreamOptions,
	wantWidth int, wantHeight int,
) browserPlaybackEvidence {
	t.Helper()
	return runBrowserLiveRTCEncoderDirectRTPPlaybackStatsWithFeedback(
		t, browserPath, label, query, options, wantWidth, wantHeight, rtcSenderFeedbackOptions{})
}

func runBrowserLiveRTCEncoderDirectRTPPlaybackStatsWithFeedback(
	t *testing.T, browserPath string, label string, query string, options rtcEncoderRTPStreamOptions,
	wantWidth int, wantHeight int, feedback rtcSenderFeedbackOptions,
) browserPlaybackEvidence {
	t.Helper()
	return runBrowserLiveRTCEncoderDirectRTPPlaybackStatsWithFeedbackFrames(
		t, browserPath, label, query, options, wantWidth, wantHeight, feedback, 45)
}

func runBrowserLiveRTCEncoderDirectRTPPlaybackStatsWithFeedbackFrames(
	t *testing.T, browserPath string, label string, query string, options rtcEncoderRTPStreamOptions,
	wantWidth int, wantHeight int, feedback rtcSenderFeedbackOptions, minFrames int,
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
		}, streamErr, options, feedback)
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
	browserCtx, closeBrowser := newBrowserE2EContext(t, ctx, browserPath)
	defer closeBrowser()

	got := browserPlaybackEvidence{}
	if err := chromedp.Run(browserCtx,
		chromedp.Navigate(server.URL+"?"+query),
		chromedp.Evaluate(browserPlaybackProbeJS(minFrames), &got, evalAwaitPromise),
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
	assertBrowserPlaybackEvidenceWithSize(t, label, got, wantWidth, wantHeight)
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
	browserCtx, closeBrowser := newBrowserE2EContext(t, ctx, browserPath)
	defer closeBrowser()

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
	browserCtx, closeBrowser := newBrowserE2EContext(t, ctx, browserPath)
	defer closeBrowser()

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
	assertBrowserPlaybackEvidenceWithSize(t, label, got, width, height)
}

func assertBrowserPlaybackEvidenceWithSize(t *testing.T, label string, got browserPlaybackEvidence, wantWidth int, wantHeight int) {
	t.Helper()
	if !got.OK {
		t.Fatalf("%s browser AV1 playback probe failed: %s; last=%+v", label, got.Error, got)
	}
	if got.VideoWidth != wantWidth || got.VideoHeight != wantHeight {
		t.Fatalf("%s browser decoded size=%dx%d want %dx%d", label, got.VideoWidth, got.VideoHeight, wantWidth, wantHeight)
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
	pc, err := newRTCEncoderRTPPeerConnectionWithOptions(streamOptions)
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
	bindBrowserReceiverEstimatedMaximumBitrateFeedback(&streamOptions, &feedback)
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
			if streamOptions.DisableTransportWideCC {
				extensions.TransportWideCCID = 0
				extensions.TransportWideCC02ID = 0
			}
			options := streamOptions
			options.HeaderExtensions = extensions
			if options.OnHeaderExtensions != nil {
				options.OnHeaderExtensions(extensions)
			}
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

func bindBrowserReceiverEstimatedMaximumBitrateFeedback(
	options *rtcEncoderRTPStreamOptions,
	feedback *rtcSenderFeedbackOptions,
) {
	if options == nil || feedback == nil || options.ReceiverEstimatedMaximumBitrate != nil {
		return
	}
	rembUpdates := make(chan goav1.RTCPReceiverEstimatedMaximumBitrate, 16)
	options.ReceiverEstimatedMaximumBitrate = rembUpdates
	prev := feedback.OnReceiverEstimatedMaximumBitrate
	feedback.OnReceiverEstimatedMaximumBitrate = func(remb *rtcp.ReceiverEstimatedMaximumBitrate) {
		if prev != nil {
			prev(remb)
		}
		update, ok := rtcReceiverEstimatedMaximumBitrateFromPion(remb)
		if !ok {
			return
		}
		select {
		case rembUpdates <- update:
		default:
		}
	}
}

func rtcSenderFeedbackTotal(counters *rtcSenderFeedbackCounters) int64 {
	if counters == nil {
		return 0
	}
	return counters.PictureLoss.Load() + counters.FullIntra.Load() + counters.NACK.Load() +
		counters.ReceiverEstimatedMaximumBitrate.Load() + counters.TransportLayerCC.Load()
}

func rtcSenderFeedbackString(counters *rtcSenderFeedbackCounters) string {
	if counters == nil {
		return "pli=0 fir=0 nack=0 remb=0 twcc=0"
	}
	return fmt.Sprintf("pli=%d fir=%d nack=%d remb=%d twcc=%d",
		counters.PictureLoss.Load(), counters.FullIntra.Load(), counters.NACK.Load(),
		counters.ReceiverEstimatedMaximumBitrate.Load(), counters.TransportLayerCC.Load())
}

func newRTCEncoderRTPPeerConnection() (*webrtc.PeerConnection, error) {
	return newRTCEncoderRTPPeerConnectionWithOptions(rtcEncoderRTPStreamOptions{})
}

func newRTCEncoderRTPPeerConnectionWithOptions(options rtcEncoderRTPStreamOptions) (*webrtc.PeerConnection, error) {
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
	if !options.DisableTransportWideCC {
		mediaEngine.RegisterFeedback(
			webrtc.RTCPFeedback{Type: webrtc.TypeRTCPFBTransportCC}, webrtc.RTPCodecTypeVideo)
		if err := mediaEngine.RegisterHeaderExtension(
			webrtc.RTPHeaderExtensionCapability{URI: goav1.AV1RTPTransportWideCCURI},
			webrtc.RTPCodecTypeVideo,
		); err != nil {
			return nil, err
		}
		if err := mediaEngine.RegisterHeaderExtension(
			webrtc.RTPHeaderExtensionCapability{URI: goav1.AV1RTPTransportWideCC02URI},
			webrtc.RTPCodecTypeVideo,
		); err != nil {
			return nil, err
		}
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

func newBrowserE2EContext(t *testing.T, parent context.Context, browserPath string) (context.Context, func()) {
	t.Helper()
	options := append([]chromedp.ExecAllocatorOption{}, chromedp.DefaultExecAllocatorOptions[:]...)
	options = append(options,
		chromedp.ExecPath(browserPath),
		chromedp.Flag("headless", "new"),
		chromedp.Flag("autoplay-policy", "no-user-gesture-required"),
		chromedp.Flag("disable-background-timer-throttling", true),
		chromedp.Flag("disable-renderer-backgrounding", true),
		chromedp.Flag("mute-audio", true),
		chromedp.NoSandbox,
		chromedp.UserDataDir(t.TempDir()),
	)
	allocCtx, cancelAlloc := chromedp.NewExecAllocator(parent, options...)
	browserCtx, cancelBrowser := chromedp.NewContext(allocCtx)
	return browserCtx, func() {
		closeCtx, cancelClose := context.WithTimeout(browserCtx, 5*time.Second)
		if err := chromedp.Cancel(closeCtx); err != nil && !errors.Is(err, context.Canceled) {
			t.Logf("browser cleanup: %v", err)
		}
		cancelClose()
		cancelBrowser()
		cancelAlloc()
	}
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
