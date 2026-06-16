// Package encoder is the home for the in-scope WebRTC realtime AV1 encoder.
//
// The encoder target is WebRTC use only, not offline/general-purpose encoding:
// bitrate, framerate, resolution, keyframe, temporal/spatial layer, SVC,
// screen-content/camera, realtime speed/quality, low-overhead OBU/RTP output,
// and dependency metadata controls. The friendly pixel encoder emits AV1
// bitstreams for 8-bit I420 L1T1/L1T2/L1T3 streams and multi-spatial
// simulcast/key-only SVC pictures through EncodePicture, with runtime
// bitrate/framerate/scalability reconfiguration. The lower-level WebRTC control
// surface validates W3C SVC mode metadata, temporal/spatial dependency
// structures, key-shift scheduling, and RTP dependency descriptors for
// caller-supplied frame payloads. Full delta-inter-layer SVC pixel coding,
// high-bit-depth input, and non-4:2:0 input remain open and should be ported
// from pinned libaom/libwebrtc behavior with speed-sensitive architecture
// checked against pinned SVT-AV1.
package encoder
