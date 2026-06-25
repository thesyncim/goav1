// Command browser-push streams realtime AV1 from the goav1 encoder to a web
// browser over WebRTC. It serves a one-page player; the browser offers a
// receive-only peer connection, the server answers and pushes an animated
// synthetic 1080p scene as an AV1 RTP track. Picture-loss feedback from the
// browser forces a keyframe, so the stream recovers from packet loss the way
// a production WebRTC sender does.
//
//	go run . -listen :8080 -bitrate 4000000
//	open http://localhost:8080
package main

import (
	_ "embed"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/pion/rtcp"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"

	goav1 "github.com/thesyncim/goav1"
)

//go:embed index.html
var indexHTML []byte

const (
	width  = 1920
	height = 1080
	fps    = 30
)

func main() {
	listen := flag.String("listen", ":8080", "HTTP listen address")
	bitrate := flag.Int("bitrate", 4_000_000, "CBR target in bits per second")
	flag.Parse()

	http.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexHTML)
	})
	http.HandleFunc("/offer", func(w http.ResponseWriter, r *http.Request) {
		if err := handleOffer(w, r, *bitrate); err != nil {
			log.Printf("offer: %v", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})
	log.Printf("serving on %s — open http://localhost%s", *listen, *listen)
	log.Fatal(http.ListenAndServe(*listen, nil))
}

// handleOffer answers one browser offer and starts a stream for it.
func handleOffer(w http.ResponseWriter, r *http.Request, bitrate int) error {
	return handleOfferWithPeerConnectionHook(w, r, bitrate, nil)
}

func handleOfferWithPeerConnectionHook(
	w http.ResponseWriter, r *http.Request, bitrate int, onPeerConnection func(*webrtc.PeerConnection),
) error {
	offerSDP, err := io.ReadAll(r.Body)
	if err != nil {
		return err
	}

	pc, err := webrtc.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		return err
	}
	if onPeerConnection != nil {
		onPeerConnection(pc)
	}
	track, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeAV1},
		"video", "goav1",
	)
	if err != nil {
		pc.Close()
		return err
	}
	sender, err := pc.AddTrack(track)
	if err != nil {
		pc.Close()
		return err
	}

	// Picture-loss feedback forces the next frame to restart the stream
	// with a keyframe.
	var wantKey atomic.Bool
	go func() {
		buf := make([]byte, 1500)
		for {
			n, _, err := sender.Read(buf)
			if err != nil {
				return
			}
			packets, err := rtcp.Unmarshal(buf[:n])
			if err != nil {
				continue
			}
			for _, p := range packets {
				switch p.(type) {
				case *rtcp.PictureLossIndication, *rtcp.FullIntraRequest, *rtcp.TransportLayerNack:
					wantKey.Store(true)
				}
			}
		}
	}()

	done := make(chan struct{})
	pc.OnConnectionStateChange(func(s webrtc.PeerConnectionState) {
		log.Printf("peer connection: %s", s)
		switch s {
		case webrtc.PeerConnectionStateConnected:
			go stream(track, &wantKey, bitrate, done)
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
		Type: webrtc.SDPTypeOffer, SDP: string(offerSDP),
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

// stream encodes the animated scene and writes one temporal unit per tick
// until the peer connection goes away.
func stream(track *webrtc.TrackLocalStaticSample, wantKey *atomic.Bool, bitrate int, done <-chan struct{}) {
	enc, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
		Width: width, Height: height,
		TargetBitrate: bitrate, Framerate: fps,
		TemporalLayers: 2,
	})
	if err != nil {
		log.Printf("encoder: %v", err)
		return
	}
	scene := newScene()
	ticker := time.NewTicker(time.Second / fps)
	defer ticker.Stop()
	frameDur := time.Second / fps
	for n := 0; ; n++ {
		select {
		case <-done:
			return
		case <-ticker.C:
		}
		out, err := enc.Encode(scene.frame(n), wantKey.Swap(false))
		if err != nil {
			log.Printf("encode: %v", err)
			return
		}
		// WriteSample packetizes synchronously, so the encoder-owned
		// temporal unit needs no copy.
		if err := track.WriteSample(media.Sample{Data: out.Data, Duration: frameDur}); err != nil {
			log.Printf("write sample: %v", err)
			return
		}
	}
}

// scene renders a panning textured background with two moving blocks - the
// same shape as the encoder's own benchmarks, cheap enough to generate at
// frame rate.
type scene struct {
	wide   []byte
	frame0 goav1.I420Frame
}

func newScene() *scene {
	rng := rand.New(rand.NewSource(7))
	wide := make([]byte, (width+512)*height)
	for y := 0; y < height; y++ {
		row := wide[y*(width+512) : (y+1)*(width+512)]
		for x := range row {
			row[x] = byte(60 + (x/7+y/9)%70 + rng.Intn(25))
		}
		for x := 1; x < len(row)-1; x++ {
			row[x] = byte((int(row[x-1]) + 2*int(row[x]) + int(row[x+1])) >> 2)
		}
	}
	cw, ch := width/2, height/2
	f := goav1.I420Frame{
		Y: make([]byte, width*height), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
		YStride: width, ChromaStride: cw, Width: width, Height: height,
	}
	for i := range f.U {
		f.U[i] = 120
		f.V[i] = 130
	}
	return &scene{wide: wide, frame0: f}
}

func (s *scene) frame(n int) goav1.I420Frame {
	f := s.frame0
	off := (n * 4) % 512
	for y := 0; y < height; y++ {
		copy(f.Y[y*width:(y+1)*width], s.wide[y*(width+512)+off:])
	}
	for _, obj := range [2][3]int{{(200 + n*6) % (width - 96), 300, 96}, {width - 100 - (n*5)%(width-164), 700, 64}} {
		ox, oy, sz := obj[0], obj[1], obj[2]
		for y := oy; y < oy+sz && y < height; y++ {
			for x := ox; x < ox+sz && x < width; x++ {
				if x >= 0 {
					f.Y[y*width+x] = 215
				}
			}
		}
	}
	return f
}
