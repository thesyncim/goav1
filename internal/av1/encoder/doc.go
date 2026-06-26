// Package encoder is the home for the in-scope WebRTC realtime AV1 encoder.
//
// The encoder target is WebRTC use only, not offline/general-purpose encoding:
// bitrate, framerate, resolution, keyframe, temporal/spatial layer, SVC,
// screen-content/camera, realtime speed/quality, low-overhead OBU/RTP output,
// and dependency metadata controls. Root-level friendly pixel APIs emit AV1
// bitstreams for 8-bit profile-0 WebRTC streams from I420/NV12/NV21 inputs plus
// I422/I444/I400 and generic Frame adapters, including native monochrome for
// explicit RTC configs across single-spatial, simulcast, and shared-reference
// SVC pictures through EncodePicture, with runtime
// bitrate/framerate/scalability reconfiguration. The lower-level WebRTC control
// surface validates W3C SVC mode metadata, temporal/spatial dependency
// structures, key-shift scheduling, and RTP dependency descriptors for
// caller-supplied frame payloads, plus sequence-matched Frame loading for
// profile-0/1/2 sample formats, including explicit sequence color config.
// Native high-bit-depth and true chroma non-4:2:0 bitstream encoding in the
// friendly pixel encoder, plus broader oracle coverage, remain open and should
// be ported from pinned libaom/libwebrtc behavior with
// speed-sensitive architecture checked against pinned SVT-AV1.
package encoder
