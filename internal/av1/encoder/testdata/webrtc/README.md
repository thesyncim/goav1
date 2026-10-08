# Pinned libwebrtc test data

These files are verbatim copies from libwebrtc commit
`7974ac00e6e0046950002bda6a38eb515dbe48a5`.

| File | Upstream path | SHA-256 | Gitiles |
| --- | --- | --- | --- |
| `scalability_mode.h` | `api/video_codecs/scalability_mode.h` | `4893ebe72987309bcf589e942fe1e3e848d6e5b90da44bbe0be4426db46afa46` | [source](https://webrtc.googlesource.com/src/+/7974ac00e6e0046950002bda6a38eb515dbe48a5/api/video_codecs/scalability_mode.h) |
| `LICENSE` | `LICENSE` | `ab00a482b6a3902e40211b43c5d0441962ea99b6cc7c25c0f243fa270b78d482` | [source](https://webrtc.googlesource.com/src/+/7974ac00e6e0046950002bda6a38eb515dbe48a5/LICENSE) |
| `PATENTS` | `PATENTS` | `01462e2068d1a04c2274f3389773014c14ed9bc3446b28303543bd3e3c064145` | [source](https://webrtc.googlesource.com/src/+/7974ac00e6e0046950002bda6a38eb515dbe48a5/PATENTS) |

`TestAppendWebRTCScalabilityModesMatchesPinnedLibWebRTC` reads the catalog
directly from this header, keeping the expected list independent of the Go
catalog implementation.
