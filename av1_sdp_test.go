package goav1_test

import (
	"errors"
	"strings"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestAV1SDPConstants(t *testing.T) {
	if av1.AV1RTPMediaType != "video/AV1" {
		t.Fatalf("AV1RTPMediaType = %q, want video/AV1", av1.AV1RTPMediaType)
	}
	if av1.AV1RTPEncodingName != "AV1" {
		t.Fatalf("AV1RTPEncodingName = %q, want AV1", av1.AV1RTPEncodingName)
	}
	if av1.AV1RTPClockRate != 90000 {
		t.Fatalf("AV1RTPClockRate = %d, want 90000", av1.AV1RTPClockRate)
	}
	if av1.AV1RTPDependencyDescriptorURI != "https://aomediacodec.github.io/av1-rtp-spec/#dependency-descriptor-rtp-header-extension" {
		t.Fatalf("AV1RTPDependencyDescriptorURI = %q", av1.AV1RTPDependencyDescriptorURI)
	}
	if av1.AV1RTPMIDURI != "urn:ietf:params:rtp-hdrext:sdes:mid" ||
		av1.AV1RTPStreamIDURI != "urn:ietf:params:rtp-hdrext:sdes:rtp-stream-id" ||
		av1.AV1RTPRepairedStreamIDURI != "urn:ietf:params:rtp-hdrext:sdes:repaired-rtp-stream-id" {
		t.Fatalf("unexpected RTP SDES extmap URI constants")
	}
	if av1.AV1SDPFmtpProfile != "profile" ||
		av1.AV1SDPFmtpLevelIdx != "level-idx" ||
		av1.AV1SDPFmtpTier != "tier" {
		t.Fatalf("unexpected AV1 SDP fmtp keys")
	}
	if av1.AV1SDPRIDMaxWidth != "max-width" ||
		av1.AV1SDPRIDMaxHeight != "max-height" ||
		av1.AV1SDPRIDMaxFrameRate != "max-fps" ||
		av1.AV1SDPRIDMaxFrameSize != "max-fs" ||
		av1.AV1SDPRIDMaxBitrate != "max-br" ||
		av1.AV1SDPRIDMaxPixelsPerSecond != "max-pps" ||
		av1.AV1SDPRIDMaxBitsPerPixel != "max-bpp" ||
		av1.AV1SDPRIDDepend != "depend" {
		t.Fatalf("unexpected AV1 SDP RID keys")
	}
	if got := av1.DefaultAV1SDPFmtpParameters(); got != (av1.AV1SDPFmtpParameters{Profile: 0, LevelIdx: 5, Tier: 0}) {
		t.Fatalf("DefaultAV1SDPFmtpParameters = %+v", got)
	}
	if av1.AV1SDPSimulcastPausedPrefix != "~" {
		t.Fatalf("AV1SDPSimulcastPausedPrefix = %q, want ~", av1.AV1SDPSimulcastPausedPrefix)
	}
}

func TestParseAV1SDPFmtp(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want av1.AV1SDPFmtpParameters
	}{
		{
			name: "empty defaults",
			in:   "",
			want: av1.AV1SDPFmtpParameters{Profile: 0, LevelIdx: 5, Tier: 0},
		},
		{
			name: "profile level tier",
			in:   "profile=2; level-idx=8; tier=1",
			want: av1.AV1SDPFmtpParameters{Profile: 2, LevelIdx: 8, Tier: 1},
		},
		{
			name: "case and spacing",
			in:   " LEVEL-IDX = 31 ; x-google-start-bitrate=800 ; PROFILE = 1 ; TIER = 1 ",
			want: av1.AV1SDPFmtpParameters{Profile: 1, LevelIdx: 31, Tier: 1},
		},
		{
			name: "profile only",
			in:   "profile=1",
			want: av1.AV1SDPFmtpParameters{Profile: 1, LevelIdx: 5, Tier: 0},
		},
		{
			name: "trailing semicolon",
			in:   "profile=0; level-idx=5; tier=0;",
			want: av1.AV1SDPFmtpParameters{Profile: 0, LevelIdx: 5, Tier: 0},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := av1.ParseAV1SDPFmtp(tc.in)
			if err != nil {
				t.Fatalf("ParseAV1SDPFmtp returned error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("ParseAV1SDPFmtp = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestAV1SDPFmtpRejectsInvalidConfig(t *testing.T) {
	for _, params := range []av1.AV1SDPFmtpParameters{
		{Profile: -1, LevelIdx: 5, Tier: 0},
		{Profile: 3, LevelIdx: 5, Tier: 0},
		{Profile: 0, LevelIdx: -1, Tier: 0},
		{Profile: 0, LevelIdx: 10, Tier: 0},
		{Profile: 0, LevelIdx: 28, Tier: 0},
		{Profile: 0, LevelIdx: 32, Tier: 0},
		{Profile: 0, LevelIdx: 5, Tier: 1},
		{Profile: 0, LevelIdx: 8, Tier: 2},
	} {
		if err := params.Validate(); !errors.Is(err, av1.ErrSDPInvalidConfig) {
			t.Fatalf("Validate(%+v) error = %v, want ErrSDPInvalidConfig", params, err)
		}
	}

	for _, in := range []string{
		"profile=3",
		"level-idx=10",
		"level-idx=28",
		"level-idx=32",
		"level-idx=5; tier=1",
		"tier=two",
		"profile=0; profile=1",
		"level-idx=5; level-idx=8",
		"tier=0; tier=1",
		"profile",
	} {
		if _, err := av1.ParseAV1SDPFmtp(in); !errors.Is(err, av1.ErrSDPInvalidConfig) {
			t.Fatalf("ParseAV1SDPFmtp(%q) error = %v, want ErrSDPInvalidConfig", in, err)
		}
	}
}

func TestAV1SDPFmtpAppendAndAllows(t *testing.T) {
	params := av1.AV1SDPFmtpParameters{Profile: 2, LevelIdx: 8, Tier: 1}
	got, err := params.Fmtp()
	if err != nil {
		t.Fatalf("Fmtp returned error: %v", err)
	}
	if got != "profile=2; level-idx=8; tier=1" {
		t.Fatalf("Fmtp = %q", got)
	}
	appended, err := params.AppendFmtp([]byte("a=fmtp:98 "))
	if err != nil {
		t.Fatalf("AppendFmtp returned error: %v", err)
	}
	if string(appended) != "a=fmtp:98 profile=2; level-idx=8; tier=1" {
		t.Fatalf("AppendFmtp = %q", appended)
	}

	caps := av1.AV1SDPFmtpParameters{Profile: 2, LevelIdx: 31, Tier: 1}
	allowed, err := caps.Allows(params)
	if err != nil {
		t.Fatalf("Allows returned error: %v", err)
	}
	if !allowed {
		t.Fatal("profile 2 level max tier 1 rejected profile 2 level 8 tier 1")
	}
	allowed, err = av1.DefaultAV1SDPFmtpParameters().Allows(params)
	if err != nil {
		t.Fatalf("Allows default returned error: %v", err)
	}
	if allowed {
		t.Fatal("default profile 0 level 5 tier 0 allowed profile 2 level 8 tier 1")
	}
	if _, err := caps.Allows(av1.AV1SDPFmtpParameters{Profile: 0, LevelIdx: 10}); !errors.Is(err, av1.ErrSDPInvalidConfig) {
		t.Fatalf("Allows invalid stream error = %v, want ErrSDPInvalidConfig", err)
	}
}

func TestParseAV1SDPExtmap(t *testing.T) {
	extmap, err := av1.ParseAV1SDPExtmap("a=extmap:4/recvonly https://aomediacodec.github.io/av1-rtp-spec/#dependency-descriptor-rtp-header-extension attrs")
	if err != nil {
		t.Fatalf("ParseAV1SDPExtmap returned error: %v", err)
	}
	if extmap.ID != 4 ||
		extmap.Direction != "recvonly" ||
		extmap.URI != av1.AV1RTPDependencyDescriptorURI ||
		extmap.Attributes != "attrs" {
		t.Fatalf("ParseAV1SDPExtmap = %+v", extmap)
	}
	line, err := extmap.SDP()
	if err != nil {
		t.Fatalf("Extmap SDP returned error: %v", err)
	}
	if line != "a=extmap:4/recvonly https://aomediacodec.github.io/av1-rtp-spec/#dependency-descriptor-rtp-header-extension attrs" {
		t.Fatalf("Extmap SDP = %q", line)
	}

	noDirection, err := av1.ParseAV1SDPExtmap("1 urn:ietf:params:rtp-hdrext:sdes:mid")
	if err != nil {
		t.Fatalf("ParseAV1SDPExtmap without prefix returned error: %v", err)
	}
	if noDirection.ID != 1 || noDirection.Direction != "" || noDirection.URI != av1.AV1RTPMIDURI {
		t.Fatalf("ParseAV1SDPExtmap no direction = %+v", noDirection)
	}
}

func TestAV1SDPExtmapRejectsInvalidConfig(t *testing.T) {
	for _, extmap := range []av1.AV1SDPExtmap{
		{},
		{ID: -1, URI: av1.AV1RTPMIDURI},
		{ID: 256, URI: av1.AV1RTPMIDURI},
		{ID: 1, Direction: "both", URI: av1.AV1RTPMIDURI},
		{ID: 1},
	} {
		if err := extmap.Validate(); !errors.Is(err, av1.ErrSDPInvalidConfig) {
			t.Fatalf("Validate(%+v) error = %v, want ErrSDPInvalidConfig", extmap, err)
		}
	}
	for _, line := range []string{
		"a=extmap:",
		"a=extmap:0 urn:ietf:params:rtp-hdrext:sdes:mid",
		"a=extmap:256 urn:ietf:params:rtp-hdrext:sdes:mid",
		"a=extmap:1/both urn:ietf:params:rtp-hdrext:sdes:mid",
		"a=extmap:one urn:ietf:params:rtp-hdrext:sdes:mid",
		"a=extmap:1",
	} {
		if _, err := av1.ParseAV1SDPExtmap(line); !errors.Is(err, av1.ErrSDPInvalidConfig) {
			t.Fatalf("ParseAV1SDPExtmap(%q) error = %v, want ErrSDPInvalidConfig", line, err)
		}
	}
}

func TestParseAV1SDPRID(t *testing.T) {
	rid, err := av1.ParseAV1SDPRID("a=rid:q recv pt=98,99;max-width=640;max-height=360;max-fps=30;max-fs=230400;max-br=700000;max-pps=6912000;max-bpp=1.25;depend=base")
	if err != nil {
		t.Fatalf("ParseAV1SDPRID returned error: %v", err)
	}
	if rid.ID != "q" || rid.Direction != av1.AV1SDPRIDDirectionReceive {
		t.Fatalf("ParseAV1SDPRID id/direction = %q/%q", rid.ID, rid.Direction)
	}
	if len(rid.PayloadTypes) != 2 || rid.PayloadTypes[0] != "98" || rid.PayloadTypes[1] != "99" {
		t.Fatalf("ParseAV1SDPRID payloads = %#v", rid.PayloadTypes)
	}
	want := av1.AV1SDPRIDRestrictions{
		MaxWidth:              640,
		MaxHeight:             360,
		MaxFrameRate:          30,
		MaxFrameSize:          230400,
		MaxBitrate:            700000,
		MaxPixelsPerSecond:    6912000,
		MaxBitsPerPixelX10000: 12500,
		DependsOn:             []string{"base"},
	}
	if rid.Restrictions.MaxWidth != want.MaxWidth ||
		rid.Restrictions.MaxHeight != want.MaxHeight ||
		rid.Restrictions.MaxFrameRate != want.MaxFrameRate ||
		rid.Restrictions.MaxFrameSize != want.MaxFrameSize ||
		rid.Restrictions.MaxBitrate != want.MaxBitrate ||
		rid.Restrictions.MaxPixelsPerSecond != want.MaxPixelsPerSecond ||
		rid.Restrictions.MaxBitsPerPixelX10000 != want.MaxBitsPerPixelX10000 ||
		len(rid.Restrictions.DependsOn) != 1 ||
		rid.Restrictions.DependsOn[0] != "base" {
		t.Fatalf("ParseAV1SDPRID restrictions = %+v, want %+v", rid.Restrictions, want)
	}
	line, err := rid.SDP()
	if err != nil {
		t.Fatalf("RID SDP returned error: %v", err)
	}
	if line != "a=rid:q recv pt=98,99;max-width=640;max-height=360;max-fps=30;max-fs=230400;max-br=700000;max-pps=6912000;max-bpp=1.25;depend=base" {
		t.Fatalf("RID SDP = %q", line)
	}

	restrictions, err := av1.ParseAV1SDPRIDRestrictions("max-width;max-height=720;max-bpp=0.0001")
	if err != nil {
		t.Fatalf("ParseAV1SDPRIDRestrictions returned error: %v", err)
	}
	if restrictions.MaxWidth != 0 ||
		restrictions.MaxHeight != 720 ||
		restrictions.MaxBitsPerPixelX10000 != 1 {
		t.Fatalf("ParseAV1SDPRIDRestrictions = %+v", restrictions)
	}
}

func TestAV1SDPRIDRejectsInvalidConfig(t *testing.T) {
	for _, restrictions := range []av1.AV1SDPRIDRestrictions{
		{MaxWidth: -1},
		{MaxBitsPerPixelX10000: -1},
		{MaxBitsPerPixelX10000: 480001},
		{DependsOn: []string{""}},
		{DependsOn: []string{"bad/id"}},
	} {
		if err := restrictions.Validate(); !errors.Is(err, av1.ErrSDPInvalidConfig) {
			t.Fatalf("Validate(%+v) error = %v, want ErrSDPInvalidConfig", restrictions, err)
		}
	}
	for _, line := range []string{
		"a=rid:q",
		"a=rid:q both max-width=640",
		"a=rid:q recv pt=",
		"a=rid:q recv pt=98,,99",
		"a=rid:q recv max-width=0",
		"a=rid:q recv max-width=-1",
		"a=rid:q recv max-width=640;max-width=641",
		"a=rid:q recv max-bpp=1",
		"a=rid:q recv max-bpp=0.0000",
		"a=rid:q recv max-bpp=48.0001",
		"a=rid:q recv max-bpp=1.12345",
		"a=rid:q recv unknown=1",
		"a=rid:q recv depend=",
		"a=rid:q recv depend=base,bad/id",
	} {
		if _, err := av1.ParseAV1SDPRID(line); !errors.Is(err, av1.ErrSDPInvalidConfig) {
			t.Fatalf("ParseAV1SDPRID(%q) error = %v, want ErrSDPInvalidConfig", line, err)
		}
	}
}

func TestAV1SDPRIDRestrictionsAllowsFrame(t *testing.T) {
	restrictions := av1.AV1SDPRIDRestrictions{
		MaxWidth:              1280,
		MaxHeight:             720,
		MaxFrameRate:          30,
		MaxFrameSize:          921600,
		MaxBitrate:            1500000,
		MaxPixelsPerSecond:    27648000,
		MaxBitsPerPixelX10000: 20000,
	}
	for _, tc := range []struct {
		name        string
		width       int
		height      int
		fps         int
		bitrate     int
		frameBits   int
		wantShape   bool
		wantEncoded bool
	}{
		{name: "within caps", width: 1280, height: 720, fps: 30, bitrate: 1500000, frameBits: 1843200, wantShape: true, wantEncoded: true},
		{name: "width too high", width: 1281, height: 720, fps: 30, bitrate: 1000, frameBits: 1000, wantShape: false, wantEncoded: false},
		{name: "fps too high", width: 1280, height: 720, fps: 31, bitrate: 1000, frameBits: 1000, wantShape: false, wantEncoded: false},
		{name: "pixel rate too high", width: 1280, height: 720, fps: 31, bitrate: 1000, frameBits: 1000, wantShape: false, wantEncoded: false},
		{name: "bitrate too high", width: 640, height: 360, fps: 30, bitrate: 1500001, frameBits: 1000, wantShape: true, wantEncoded: false},
		{name: "bpp too high", width: 640, height: 360, fps: 30, bitrate: 1000, frameBits: 460801, wantShape: true, wantEncoded: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gotShape, err := restrictions.AllowsFrame(tc.width, tc.height, tc.fps)
			if err != nil {
				t.Fatalf("AllowsFrame returned error: %v", err)
			}
			if gotShape != tc.wantShape {
				t.Fatalf("AllowsFrame = %t, want %t", gotShape, tc.wantShape)
			}
			gotEncoded, err := restrictions.AllowsEncodedFrame(tc.width, tc.height, tc.fps, tc.bitrate, tc.frameBits)
			if err != nil {
				t.Fatalf("AllowsEncodedFrame returned error: %v", err)
			}
			if gotEncoded != tc.wantEncoded {
				t.Fatalf("AllowsEncodedFrame = %t, want %t", gotEncoded, tc.wantEncoded)
			}
		})
	}
	if _, err := restrictions.AllowsFrame(640, 360, 0); !errors.Is(err, av1.ErrSDPInvalidConfig) {
		t.Fatalf("AllowsFrame invalid fps error = %v, want ErrSDPInvalidConfig", err)
	}
	if _, err := restrictions.AllowsEncodedFrame(640, 360, 30, 0, 1); !errors.Is(err, av1.ErrSDPInvalidConfig) {
		t.Fatalf("AllowsEncodedFrame invalid bitrate error = %v, want ErrSDPInvalidConfig", err)
	}
	if _, err := restrictions.AllowsEncodedFrame(640, 360, 30, 1, 0); !errors.Is(err, av1.ErrSDPInvalidConfig) {
		t.Fatalf("AllowsEncodedFrame invalid frame bits error = %v, want ErrSDPInvalidConfig", err)
	}
}

func TestParseAV1SDPSimulcast(t *testing.T) {
	simulcast, err := av1.ParseAV1SDPSimulcast("a=simulcast:send q;h,f recv ~low,mid;high")
	if err != nil {
		t.Fatalf("ParseAV1SDPSimulcast returned error: %v", err)
	}
	if len(simulcast.Send) != 2 ||
		len(simulcast.Send[0]) != 1 ||
		simulcast.Send[0][0].RID != "q" ||
		len(simulcast.Send[1]) != 2 ||
		simulcast.Send[1][0].RID != "h" ||
		simulcast.Send[1][1].RID != "f" {
		t.Fatalf("unexpected send simulcast streams: %#v", simulcast.Send)
	}
	if len(simulcast.Receive) != 2 ||
		len(simulcast.Receive[0]) != 2 ||
		!simulcast.Receive[0][0].Paused ||
		simulcast.Receive[0][0].RID != "low" ||
		simulcast.Receive[0][1].RID != "mid" ||
		simulcast.Receive[1][0].RID != "high" {
		t.Fatalf("unexpected recv simulcast streams: %#v", simulcast.Receive)
	}
	line, err := simulcast.SDP()
	if err != nil {
		t.Fatalf("Simulcast SDP returned error: %v", err)
	}
	if line != "a=simulcast:send q;h,f recv ~low,mid;high" {
		t.Fatalf("Simulcast SDP = %q", line)
	}

	reordered, err := av1.ParseAV1SDPSimulcast("recv r send s")
	if err != nil {
		t.Fatalf("ParseAV1SDPSimulcast reordered returned error: %v", err)
	}
	line, err = reordered.SDP()
	if err != nil {
		t.Fatalf("reordered SDP returned error: %v", err)
	}
	if line != "a=simulcast:send s recv r" {
		t.Fatalf("reordered SDP = %q", line)
	}
}

func TestAV1SDPSimulcastRejectsInvalidConfig(t *testing.T) {
	for _, simulcast := range []av1.AV1SDPSimulcast{
		{},
		{Send: []av1.AV1SDPSimulcastStream{{}}},
		{Send: []av1.AV1SDPSimulcastStream{{{RID: ""}}}},
		{Send: []av1.AV1SDPSimulcastStream{{{RID: "q"}, {RID: "q"}}}},
	} {
		if err := simulcast.Validate(); !errors.Is(err, av1.ErrSDPInvalidConfig) {
			t.Fatalf("Validate(%+v) error = %v, want ErrSDPInvalidConfig", simulcast, err)
		}
	}
	for _, line := range []string{
		"a=simulcast:",
		"a=simulcast:send",
		"a=simulcast:send q recv",
		"a=simulcast:both q",
		"a=simulcast:send q send h",
		"a=simulcast:send q;",
		"a=simulcast:send q,,h",
		"a=simulcast:send ~",
		"a=simulcast:send q/h",
		"a=simulcast:send q recv h send f",
	} {
		if _, err := av1.ParseAV1SDPSimulcast(line); !errors.Is(err, av1.ErrSDPInvalidConfig) {
			t.Fatalf("ParseAV1SDPSimulcast(%q) error = %v, want ErrSDPInvalidConfig", line, err)
		}
	}
}

func TestAV1SDPFmtpParametersForSequence(t *testing.T) {
	seq := av1.SequenceHeader{
		SeqProfile:           2,
		OperatingPointsCount: 3,
	}
	seq.OperatingPoints[0] = av1.OperatingPoint{SeqLevelIdx: 5, SeqTier: 0}
	seq.OperatingPoints[1] = av1.OperatingPoint{SeqLevelIdx: 8, SeqTier: 1}
	seq.OperatingPoints[2] = av1.OperatingPoint{SeqLevelIdx: 4, SeqTier: 0}
	got, err := av1.AV1SDPFmtpParametersForSequence(seq)
	if err != nil {
		t.Fatalf("AV1SDPFmtpParametersForSequence returned error: %v", err)
	}
	want := av1.AV1SDPFmtpParameters{Profile: 2, LevelIdx: 8, Tier: 1}
	if got != want {
		t.Fatalf("AV1SDPFmtpParametersForSequence = %+v, want %+v", got, want)
	}

	allowed, err := (av1.AV1SDPFmtpParameters{Profile: 2, LevelIdx: 31, Tier: 1}).AllowsSequence(seq)
	if err != nil {
		t.Fatalf("AllowsSequence returned error: %v", err)
	}
	if !allowed {
		t.Fatal("max-level receiver rejected sequence")
	}
	allowed, err = av1.DefaultAV1SDPFmtpParameters().AllowsSequence(seq)
	if err != nil {
		t.Fatalf("AllowsSequence default returned error: %v", err)
	}
	if allowed {
		t.Fatal("default receiver allowed profile 2 level 8 tier 1 sequence")
	}

	bad := seq
	bad.OperatingPointsCount = 0
	if _, err := av1.AV1SDPFmtpParametersForSequence(bad); !errors.Is(err, av1.ErrSDPInvalidConfig) {
		t.Fatalf("zero operating points error = %v, want ErrSDPInvalidConfig", err)
	}
	bad = seq
	bad.OperatingPoints[0].SeqLevelIdx = 10
	if _, err := av1.AV1SDPFmtpParametersForSequence(bad); !errors.Is(err, av1.ErrSDPInvalidConfig) {
		t.Fatalf("reserved level error = %v, want ErrSDPInvalidConfig", err)
	}
}

func TestAV1SDPSectionScanning(t *testing.T) {
	sdp := joinAV1SDPLines(
		"m=audio 9 UDP/TLS/RTP/SAVPF 111",
		"a=rtpmap:111 opus/48000/2",
		"m=video 9 UDP/TLS/RTP/SAVPF 96 98",
		"a=rtpmap:96 VP8/90000",
		"a=rtpmap:98 AV1/90000",
		"a=fmtp:98 profile=2; level-idx=8; tier=1;",
	)
	if !av1.AV1SDPNegotiates(sdp) {
		t.Fatal("AV1SDPNegotiates rejected active AV1 video section")
	}
	if !av1.AV1SDPOffersReceive(sdp) {
		t.Fatal("AV1SDPOffersReceive rejected sendrecv AV1 video section")
	}
	if !av1.AV1SDPAnswersSend(sdp) {
		t.Fatal("AV1SDPAnswersSend rejected sendrecv AV1 video section")
	}
	if !av1.AV1SDPNegotiatesParams(sdp, av1.AV1SDPFmtpParameters{Profile: 2, LevelIdx: 8, Tier: 1}) {
		t.Fatal("AV1SDPNegotiatesParams rejected matching stream")
	}
	if av1.AV1SDPOffersReceiveParams(sdp, av1.AV1SDPFmtpParameters{Profile: 2, LevelIdx: 12, Tier: 1}) {
		t.Fatal("AV1SDPOffersReceiveParams allowed stream above advertised level")
	}

	for _, tc := range []struct {
		name string
		sdp  string
	}{
		{
			name: "wrong clock",
			sdp: joinAV1SDPLines(
				"m=video 9 UDP/TLS/RTP/SAVPF 98",
				"a=rtpmap:98 AV1/48000",
			),
		},
		{
			name: "payload not in m line",
			sdp: joinAV1SDPLines(
				"m=video 9 UDP/TLS/RTP/SAVPF 97",
				"a=rtpmap:98 AV1/90000",
			),
		},
		{
			name: "port zero",
			sdp: joinAV1SDPLines(
				"m=video 0 UDP/TLS/RTP/SAVPF 98",
				"a=rtpmap:98 AV1/90000",
			),
		},
		{
			name: "inactive",
			sdp: joinAV1SDPLines(
				"m=video 9 UDP/TLS/RTP/SAVPF 98",
				"a=inactive",
				"a=rtpmap:98 AV1/90000",
			),
		},
		{
			name: "invalid fmtp",
			sdp: joinAV1SDPLines(
				"m=video 9 UDP/TLS/RTP/SAVPF 98",
				"a=rtpmap:98 AV1/90000",
				"a=fmtp:98 level-idx=10",
			),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if av1.AV1SDPNegotiates(tc.sdp) {
				t.Fatal("AV1SDPNegotiates accepted invalid SDP")
			}
		})
	}

	sendOnly := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 98",
		"a=sendonly",
		"a=rtpmap:98 AV1/90000",
	)
	if av1.AV1SDPOffersReceive(sendOnly) {
		t.Fatal("sendonly offer reported receive support")
	}
	if !av1.AV1SDPAnswersSend(sendOnly) {
		t.Fatal("sendonly answer did not report send support")
	}

	recvOnly := joinAV1SDPLines(
		"a=recvonly",
		"m=video 9 UDP/TLS/RTP/SAVPF 98",
		"a=rtpmap:98 AV1/90000",
	)
	if !av1.AV1SDPOffersReceive(recvOnly) {
		t.Fatal("session-level recvonly offer did not report receive support")
	}
	if av1.AV1SDPAnswersSend(recvOnly) {
		t.Fatal("recvonly SDP reported send support")
	}
}

func TestAV1SDPRTCPFeedbackScanning(t *testing.T) {
	sdp := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 96 98 99",
		"a=rtpmap:96 VP8/90000",
		"a=rtpmap:98 AV1/90000",
		"a=rtpmap:99 AV1/90000",
		"a=rtcp-fb:96 ccm lrr",
		"a=rtcp-fb:98 nack",
		"a=rtcp-fb:98 nack pli",
		"a=rtcp-fb:* ccm fir",
		"a=rtcp-fb:99 ccm lrr",
	)
	if !av1.AV1SDPNegotiatesRTCPFeedback(sdp, av1.AV1SDPRTCPFeedbackNACK) {
		t.Fatal("AV1SDPNegotiatesRTCPFeedback rejected payload-specific nack")
	}
	if !av1.AV1SDPNegotiatesRTCPFeedback(sdp, av1.AV1SDPRTCPFeedbackPLI) {
		t.Fatal("AV1SDPNegotiatesRTCPFeedback rejected payload-specific pli")
	}
	if !av1.AV1SDPNegotiatesRTCPFeedback(sdp, av1.AV1SDPRTCPFeedbackFIR) {
		t.Fatal("AV1SDPNegotiatesRTCPFeedback rejected wildcard fir")
	}
	if !av1.AV1SDPOffersReceiveRTCPFeedback(sdp, av1.AV1SDPRTCPFeedbackLRR) {
		t.Fatal("AV1SDPOffersReceiveRTCPFeedback rejected AV1 payload-specific lrr")
	}
	if av1.AV1SDPOffersReceiveRTCPFeedback(sdp, "goog-remb") {
		t.Fatal("AV1SDPOffersReceiveRTCPFeedback accepted missing feedback")
	}
	if av1.AV1SDPOffersReceiveRTCPFeedback(sdp, "") {
		t.Fatal("AV1SDPOffersReceiveRTCPFeedback accepted empty feedback")
	}

	wrongPayload := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 96 98",
		"a=rtpmap:96 VP8/90000",
		"a=rtpmap:98 AV1/90000",
		"a=rtcp-fb:96 ccm lrr",
	)
	if av1.AV1SDPNegotiatesRTCPFeedback(wrongPayload, av1.AV1SDPRTCPFeedbackLRR) {
		t.Fatal("AV1SDPNegotiatesRTCPFeedback accepted feedback on non-AV1 payload")
	}

	inactive := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 98",
		"a=inactive",
		"a=rtpmap:98 AV1/90000",
		"a=rtcp-fb:98 ccm lrr",
	)
	if av1.AV1SDPNegotiatesRTCPFeedback(inactive, av1.AV1SDPRTCPFeedbackLRR) {
		t.Fatal("AV1SDPNegotiatesRTCPFeedback accepted inactive section")
	}

	sendOnly := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 98",
		"a=sendonly",
		"a=rtpmap:98 AV1/90000",
		"a=rtcp-fb:98 ccm lrr",
	)
	if av1.AV1SDPOffersReceiveRTCPFeedback(sendOnly, av1.AV1SDPRTCPFeedbackLRR) {
		t.Fatal("AV1SDPOffersReceiveRTCPFeedback accepted sendonly section")
	}
	if !av1.AV1SDPAnswersSendRTCPFeedback(sendOnly, av1.AV1SDPRTCPFeedbackLRR) {
		t.Fatal("AV1SDPAnswersSendRTCPFeedback rejected sendonly section")
	}

	wildcardBeforeRTPMap := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 98",
		"a=rtcp-fb:* ccm lrr",
		"a=rtpmap:98 AV1/90000",
	)
	if !av1.AV1SDPNegotiatesRTCPFeedback(wildcardBeforeRTPMap, " CCM LRR ") {
		t.Fatal("AV1SDPNegotiatesRTCPFeedback rejected normalized wildcard feedback")
	}
}

func TestAV1SDPRTCPFeedbackSDP(t *testing.T) {
	pli := av1.AV1SDPRTCPFeedback{
		PayloadType: "98",
		Feedback:    " NACK   PLI ",
	}
	line, err := pli.SDP()
	if err != nil {
		t.Fatalf("PLI SDP: %v", err)
	}
	if line != "a=rtcp-fb:98 nack pli" {
		t.Fatalf("PLI SDP line=%q", line)
	}
	prefix := []byte("x")
	appended, err := (av1.AV1SDPRTCPFeedback{
		PayloadType: "*",
		Feedback:    av1.AV1SDPRTCPFeedbackLRR,
	}).AppendSDP(prefix)
	if err != nil {
		t.Fatalf("wildcard LRR AppendSDP: %v", err)
	}
	if string(appended) != "xa=rtcp-fb:* ccm lrr" {
		t.Fatalf("wildcard LRR line=%q", appended)
	}

	sdp := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 98",
		"a=rtpmap:98 AV1/90000",
		line,
		string(appended[1:]),
	)
	if !av1.AV1SDPNegotiatesRTCPFeedback(sdp, av1.AV1SDPRTCPFeedbackPLI) {
		t.Fatal("generated PLI rtcp-fb line did not negotiate")
	}
	if !av1.AV1SDPNegotiatesRTCPFeedback(sdp, av1.AV1SDPRTCPFeedbackLRR) {
		t.Fatal("generated wildcard LRR rtcp-fb line did not negotiate")
	}
}

func TestAV1SDPRTCPFeedbackRejectsInvalid(t *testing.T) {
	tests := []av1.AV1SDPRTCPFeedback{
		{PayloadType: "", Feedback: av1.AV1SDPRTCPFeedbackPLI},
		{PayloadType: "av1", Feedback: av1.AV1SDPRTCPFeedbackPLI},
		{PayloadType: "128", Feedback: av1.AV1SDPRTCPFeedbackPLI},
		{PayloadType: "98\n99", Feedback: av1.AV1SDPRTCPFeedbackPLI},
		{PayloadType: "98", Feedback: ""},
		{PayloadType: "98", Feedback: "nack\npli"},
	}
	for _, tc := range tests {
		if err := tc.Validate(); !errors.Is(err, av1.ErrSDPInvalidConfig) {
			t.Fatalf("Validate(%+v) err=%v want %v", tc, err, av1.ErrSDPInvalidConfig)
		}
		if out, err := tc.AppendSDP([]byte("prefix")); !errors.Is(err, av1.ErrSDPInvalidConfig) || string(out) != "prefix" {
			t.Fatalf("AppendSDP(%+v) out=%q err=%v", tc, out, err)
		}
		if line, err := tc.SDP(); !errors.Is(err, av1.ErrSDPInvalidConfig) || line != "" {
			t.Fatalf("SDP(%+v) line=%q err=%v", tc, line, err)
		}
	}
}

func TestAV1SDPHeaderExtensionScanning(t *testing.T) {
	sdp := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 96 98",
		"a=rtpmap:96 VP8/90000",
		"a=rtpmap:98 AV1/90000",
		"a=extmap:1 urn:ietf:params:rtp-hdrext:sdes:mid",
		"a=extmap:2/recvonly urn:ietf:params:rtp-hdrext:sdes:rtp-stream-id",
		"a=extmap:3/sendonly urn:ietf:params:rtp-hdrext:sdes:repaired-rtp-stream-id",
		"a=extmap:4/sendrecv https://aomediacodec.github.io/av1-rtp-spec/#dependency-descriptor-rtp-header-extension",
	)
	if !av1.AV1SDPNegotiatesHeaderExtension(sdp, av1.AV1RTPDependencyDescriptorURI) {
		t.Fatal("AV1SDPNegotiatesHeaderExtension rejected dependency descriptor")
	}
	if !av1.AV1SDPOffersReceiveHeaderExtension(sdp, av1.AV1RTPStreamIDURI) {
		t.Fatal("AV1SDPOffersReceiveHeaderExtension rejected recvonly RID")
	}
	if av1.AV1SDPAnswersSendHeaderExtension(sdp, av1.AV1RTPStreamIDURI) {
		t.Fatal("AV1SDPAnswersSendHeaderExtension accepted recvonly RID")
	}
	if !av1.AV1SDPAnswersSendHeaderExtension(sdp, av1.AV1RTPRepairedStreamIDURI) {
		t.Fatal("AV1SDPAnswersSendHeaderExtension rejected sendonly repaired RID")
	}
	if !av1.AV1SDPOffersReceiveHeaderExtension(sdp, av1.AV1RTPMIDURI) {
		t.Fatal("AV1SDPOffersReceiveHeaderExtension rejected inherited MID")
	}
	if av1.AV1SDPOffersReceiveHeaderExtension(sdp, "") {
		t.Fatal("AV1SDPOffersReceiveHeaderExtension accepted empty URI")
	}

	sendOnlySection := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 98",
		"a=sendonly",
		"a=rtpmap:98 AV1/90000",
		"a=extmap:1 urn:ietf:params:rtp-hdrext:sdes:mid",
	)
	if av1.AV1SDPOffersReceiveHeaderExtension(sendOnlySection, av1.AV1RTPMIDURI) {
		t.Fatal("sendonly inherited extmap reported receive support")
	}
	if !av1.AV1SDPAnswersSendHeaderExtension(sendOnlySection, av1.AV1RTPMIDURI) {
		t.Fatal("sendonly inherited extmap did not report send support")
	}

	wrongMedia := joinAV1SDPLines(
		"m=audio 9 UDP/TLS/RTP/SAVPF 111",
		"a=rtpmap:111 opus/48000/2",
		"a=extmap:1 https://aomediacodec.github.io/av1-rtp-spec/#dependency-descriptor-rtp-header-extension",
	)
	if av1.AV1SDPNegotiatesHeaderExtension(wrongMedia, av1.AV1RTPDependencyDescriptorURI) {
		t.Fatal("audio extmap reported AV1 header-extension support")
	}
}

func TestAV1SDPRIDFrameScanning(t *testing.T) {
	sdp := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 96 98 99",
		"a=rtpmap:96 VP8/90000",
		"a=rtpmap:98 AV1/90000",
		"a=rtpmap:99 AV1/90000",
		"a=rid:v recv pt=96;max-width=320;max-height=180;max-fps=15",
		"a=rid:q recv pt=98;max-width=640;max-height=360;max-fps=30;max-fs=230400;max-pps=6912000",
		"a=rid:h recv pt=99;max-width=1280;max-height=720;max-fps=30",
		"a=rid:out send pt=99;max-width=1920;max-height=1080;max-fps=60",
	)
	if !av1.AV1SDPOffersReceiveFrame(sdp, 640, 360, 30) {
		t.Fatal("AV1SDPOffersReceiveFrame rejected matching AV1 recv RID")
	}
	if av1.AV1SDPOffersReceiveFrame(sdp, 1280, 720, 31) {
		t.Fatal("AV1SDPOffersReceiveFrame accepted frame above recv RID fps")
	}
	if !av1.AV1SDPAnswersSendFrame(sdp, 1920, 1080, 60) {
		t.Fatal("AV1SDPAnswersSendFrame rejected matching AV1 send RID")
	}

	wrongPayloadRIDOnly := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 96 98",
		"a=rtpmap:96 VP8/90000",
		"a=rtpmap:98 AV1/90000",
		"a=rid:v recv pt=96;max-width=320;max-height=180;max-fps=15",
	)
	if !av1.AV1SDPOffersReceiveFrame(wrongPayloadRIDOnly, 1920, 1080, 60) {
		t.Fatal("non-AV1 RID restrictions constrained AV1 payload")
	}

	unscopedRID := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 98",
		"a=rtpmap:98 AV1/90000",
		"a=rid:q recv max-width=640;max-height=360;max-fps=30",
	)
	if !av1.AV1SDPOffersReceiveFrame(unscopedRID, 640, 360, 30) {
		t.Fatal("AV1SDPOffersReceiveFrame rejected unscoped matching RID")
	}
	if av1.AV1SDPOffersReceiveFrame(unscopedRID, 1280, 720, 30) {
		t.Fatal("AV1SDPOffersReceiveFrame accepted frame above unscoped RID")
	}

	duplicateRID := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 98",
		"a=rtpmap:98 AV1/90000",
		"a=rid:q recv max-width=640;max-height=360;max-fps=30",
		"a=rid:q recv max-width=1280;max-height=720;max-fps=30",
	)
	if !av1.AV1SDPOffersReceiveFrame(duplicateRID, 1280, 720, 30) {
		t.Fatal("duplicate RID was not discarded before frame matching")
	}

	invalidRID := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 98",
		"a=rtpmap:98 AV1/90000",
		"a=rid:q recv max-width=wide",
	)
	if !av1.AV1SDPOffersReceiveFrame(invalidRID, 1280, 720, 30) {
		t.Fatal("invalid RID line was not discarded before frame matching")
	}

	sendOnly := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 98",
		"a=sendonly",
		"a=rtpmap:98 AV1/90000",
		"a=rid:q recv max-width=640;max-height=360;max-fps=30",
	)
	if av1.AV1SDPOffersReceiveFrame(sendOnly, 640, 360, 30) {
		t.Fatal("sendonly section reported receive frame support")
	}
}

func TestAV1SDPSimulcastScanning(t *testing.T) {
	sdp := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 96 98 99 100",
		"a=rtpmap:96 VP8/90000",
		"a=rtpmap:98 AV1/90000",
		"a=rtpmap:99 AV1/90000",
		"a=rtpmap:100 AV1/90000",
		"a=rid:v recv pt=96;max-width=320;max-height=180;max-fps=15",
		"a=rid:q recv pt=98;max-width=640;max-height=360;max-fps=30",
		"a=rid:h recv pt=99;max-width=1280;max-height=720;max-fps=30",
		"a=rid:f recv pt=100;max-width=1920;max-height=1080;max-fps=30",
		"a=rid:sendh send pt=99;max-width=1280;max-height=720;max-fps=30",
		"a=simulcast:recv q;h send sendh",
	)
	if !av1.AV1SDPNegotiatesSimulcast(sdp) {
		t.Fatal("AV1SDPNegotiatesSimulcast rejected active AV1 simulcast")
	}
	if !av1.AV1SDPOffersReceiveSimulcast(sdp) {
		t.Fatal("AV1SDPOffersReceiveSimulcast rejected AV1 recv simulcast")
	}
	if !av1.AV1SDPAnswersSendSimulcast(sdp) {
		t.Fatal("AV1SDPAnswersSendSimulcast rejected AV1 send simulcast")
	}
	if !av1.AV1SDPOffersReceiveFrame(sdp, 1280, 720, 30) {
		t.Fatal("AV1SDPOffersReceiveFrame rejected matching simulcast RID")
	}
	if av1.AV1SDPOffersReceiveFrame(sdp, 1920, 1080, 30) {
		t.Fatal("AV1SDPOffersReceiveFrame used RID outside simulcast recv list")
	}
	if !av1.AV1SDPAnswersSendFrame(sdp, 1280, 720, 30) {
		t.Fatal("AV1SDPAnswersSendFrame rejected matching simulcast send RID")
	}

	pausedAlternative := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 98",
		"a=rtpmap:98 AV1/90000",
		"a=rid:q recv pt=98;max-width=640;max-height=360;max-fps=30",
		"a=simulcast:recv ~q",
	)
	if !av1.AV1SDPOffersReceiveSimulcast(pausedAlternative) {
		t.Fatal("paused simulcast RID was not treated as negotiated")
	}
	if !av1.AV1SDPOffersReceiveFrame(pausedAlternative, 640, 360, 30) {
		t.Fatal("paused simulcast RID restrictions did not allow matching frame")
	}

	wrongPayload := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 96 98",
		"a=rtpmap:96 VP8/90000",
		"a=rtpmap:98 AV1/90000",
		"a=rid:v recv pt=96;max-width=320;max-height=180;max-fps=15",
		"a=simulcast:recv v",
	)
	if av1.AV1SDPOffersReceiveSimulcast(wrongPayload) {
		t.Fatal("AV1SDPOffersReceiveSimulcast accepted non-AV1 RID")
	}

	duplicateSimulcast := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 98",
		"a=rtpmap:98 AV1/90000",
		"a=rid:q recv pt=98;max-width=640;max-height=360;max-fps=30",
		"a=simulcast:recv q",
		"a=simulcast:recv q",
	)
	if av1.AV1SDPOffersReceiveSimulcast(duplicateSimulcast) {
		t.Fatal("duplicate simulcast attribute was not disabled")
	}
	if av1.AV1SDPOffersReceiveFrame(duplicateSimulcast, 1280, 720, 30) {
		t.Fatal("duplicate simulcast attribute ignored RID receiver restriction")
	}
	if !av1.AV1SDPOffersReceiveFrame(duplicateSimulcast, 640, 360, 30) {
		t.Fatal("duplicate simulcast attribute should still allow matching RID restriction")
	}

	missingRID := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 98",
		"a=rtpmap:98 AV1/90000",
		"a=simulcast:recv q",
	)
	if av1.AV1SDPOffersReceiveSimulcast(missingRID) {
		t.Fatal("AV1SDPOffersReceiveSimulcast accepted missing RID")
	}
	if av1.AV1SDPOffersReceiveFrame(missingRID, 640, 360, 30) {
		t.Fatal("AV1SDPOffersReceiveFrame accepted simulcast RID without restrictions")
	}
}

func TestAV1SDPOffersReceiveSequence(t *testing.T) {
	seq := av1.SequenceHeader{
		SeqProfile:           2,
		OperatingPointsCount: 1,
	}
	seq.OperatingPoints[0] = av1.OperatingPoint{SeqLevelIdx: 8, SeqTier: 1}
	sdp := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 98",
		"a=rtpmap:98 AV1/90000",
		"a=fmtp:98 profile=2; level-idx=8; tier=1",
	)
	if !av1.AV1SDPOffersReceiveSequence(sdp, seq) {
		t.Fatal("AV1SDPOffersReceiveSequence rejected matching sequence")
	}
	limited := joinAV1SDPLines(
		"m=video 9 UDP/TLS/RTP/SAVPF 98",
		"a=rtpmap:98 AV1/90000",
		"a=fmtp:98 profile=0; level-idx=5",
	)
	if av1.AV1SDPOffersReceiveSequence(limited, seq) {
		t.Fatal("AV1SDPOffersReceiveSequence allowed profile/level above offer")
	}
}

func joinAV1SDPLines(lines ...string) string {
	return strings.Join(lines, "\r\n")
}
