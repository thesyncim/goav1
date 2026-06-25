# browser-push — realtime AV1 to a browser over WebRTC

Streams an animated 1080p30 scene from the goav1 encoder to a browser
`<video>` element as AV1 RTP. The browser sends a receive-only offer, the
server answers, and loss feedback (PLI/FIR/NACK) forces keyframes so the stream
recovers from packet loss. The direct-RTP browser gate also proves Generic NACK
repair with a sender-side retransmission cache. The required browser gate
live-tests every `EncoderWebRTCScalabilityModes()` value: L1 modes are sent
directly, shared-reference L2/L3 SVC modes forward the base spatial layer, and
S2/S3 simulcast modes forward the highest spatial layer. The required gate also
repeats representative direct-RTP browser playback sessions across those
delivery shapes and live-tests playback while framerate, bitrate, rate control,
content hint, and representative scalability settings are reconfigured.

The example carries its own `go.mod` (with a `replace` to the repository
root), so its WebRTC dependencies stay out of the main module.

## Run

```
cd examples/browser-push
go run . -listen :8080 -bitrate 4000000
```

Open http://localhost:8080 in a browser with AV1 decode support (Chrome,
Edge, Firefox, Safari 17+).

## Shape

- `goav1.VideoEncoder` at 1080p30 CBR with two temporal layers
- Pion WebRTC v4 `TrackLocalStaticSample` with the AV1 payloader
- One goroutine per viewer; the encoder's zero-allocation steady state
  means each stream costs only its compute
- The direct-RTP test path uses dependency descriptors, exact active
  decode-target masks, and filtered spatial forwarding for browser-compatible
  multi-spatial delivery. It does not claim that Chrome can consume all
  spatial layers mixed on one static AV1 RTP track.
- The repeated direct-RTP soak path opens fresh browser sessions for L1 temporal
  layering, shared-reference SVC base-layer forwarding, and highest-layer
  simulcast forwarding.
- The live control-churn path proves those browser sessions keep decoding while
  sender settings change across framerate, bitrate, CBR/CQP, content hint, and
  single-spatial scalability modes.
