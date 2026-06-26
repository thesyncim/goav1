// Package encoder is the home for the in-scope WebRTC realtime AV1 encoder.
//
// The encoder target is WebRTC use only, not offline/general-purpose encoding:
// bitrate, framerate, resolution, keyframe, temporal/spatial layer, SVC,
// screen-content/camera, realtime speed/quality, low-overhead OBU/RTP output,
// and dependency metadata controls. Root-level friendly pixel APIs emit AV1
// bitstreams for 8-bit profile-0 WebRTC streams from I420/NV12/NV21 inputs plus
// I422/I444/I400 and generic Frame adapters, including native 8-bit profile-2
// 4:2:2, native 8-bit profile-1 4:4:4, native 8-bit monochrome, and native
// 10/12-bit monochrome RTC output for explicit single-spatial, simulcast, and
// shared-reference SVC configs, with runtime bitrate/framerate/scalability
// reconfiguration. The lower-level WebRTC control surface validates W3C SVC
// mode metadata, temporal/spatial dependency structures, key-shift scheduling,
// and RTP dependency descriptors for caller-supplied frame payloads, plus
// sequence-matched Frame loading for profile-0/1/2 sample formats, including
// explicit sequence color config. Native high-bit-depth non-monochrome color,
// broader oracle coverage, and speed-sensitive architecture remain open and
// should be checked against pinned libaom/libwebrtc/SVT-AV1 behavior.
package encoder
