// Package encoder is the home for the in-scope WebRTC realtime AV1 encoder.
//
// The encoder target is WebRTC use only, not offline/general-purpose encoding:
// bitrate, framerate, resolution, keyframe, temporal/spatial layer, SVC,
// screen-content/camera, realtime speed/quality, low-overhead OBU/RTP output,
// and dependency metadata controls. Implementation should be ported from pinned
// libaom/libwebrtc behavior with oracle coverage and zero-allocation hot paths.
package encoder
