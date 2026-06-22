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
	if av1.AV1SDPFmtpProfile != "profile" ||
		av1.AV1SDPFmtpLevelIdx != "level-idx" ||
		av1.AV1SDPFmtpTier != "tier" {
		t.Fatalf("unexpected AV1 SDP fmtp keys")
	}
	if got := av1.DefaultAV1SDPFmtpParameters(); got != (av1.AV1SDPFmtpParameters{Profile: 0, LevelIdx: 5, Tier: 0}) {
		t.Fatalf("DefaultAV1SDPFmtpParameters = %+v", got)
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
