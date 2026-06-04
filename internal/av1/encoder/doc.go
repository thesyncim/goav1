// Package encoder is the home for the in-scope WebRTC realtime AV1 encoder.
//
// The encoder target is WebRTC use only, not offline/general-purpose encoding:
// bitrate, framerate, resolution, keyframe, temporal/spatial layer, SVC,
// screen-content/camera, realtime speed/quality, low-overhead OBU/RTP output,
// and dependency metadata controls. The current package starts with the
// WebRTC/libaom-shaped control plane and temporal-unit validation; bitstream
// emission should be ported from pinned libaom/libwebrtc behavior, with
// speed-sensitive architecture checked against pinned SVT-AV1 and upstream C
// type widths preserved in new/touched parity paths. Oracle coverage and
// zero-allocation hot paths are required before usable encoder claims.
package encoder
