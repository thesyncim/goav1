package encoder

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math/rand"
	"testing"
)

// TestPFrameReuseOptimizationsPreserveEncodedOutput pins the AV1 bytes and
// reconstructed planes produced by the pre-optimization encoder. Its serial
// and multi-thread cases cover both fused and split P-frame residual paths.
func TestPFrameReuseOptimizationsPreserveEncodedOutput(t *testing.T) {
	const width, height, frameCount = 640, 360, 48
	tests := []struct {
		threads int
		stream  string
		recon   string
	}{
		{1, "25756cb8a7a842454b9f7e5fc21538b8688ca008457f4faeec5fd94d1acdff32", "9fc2e8c250e7236de86750dc0661be68f8914df5380fe540bdffc3632d2a6f34"},
		{4, "e01a86628f3689010262348c3fd6521dc0ecd385c34d107f8dfc873d945444b6", "9b6f88a516d76adbc2bb8b987783c5732359f7f610ea6aee561839b7d9e468d1"},
	}

	rng := rand.New(rand.NewSource(3))
	background := make([]byte, width*height)
	for i := range background {
		background[i] = uint8(60 + rng.Intn(60))
	}
	makeFrame := func(i int) SourceFrame420 {
		f := SourceFrame420{
			Y:            append([]byte(nil), background...),
			U:            make([]byte, width/2*(height/2)),
			V:            make([]byte, width/2*(height/2)),
			YStride:      width,
			ChromaStride: width / 2,
			Width:        width,
			Height:       height,
		}
		for i := range f.U {
			f.U[i], f.V[i] = 120, 130
		}
		sx, sy := (i*4)%(width-32), (i*2)%(height-32)
		for y := sy; y < sy+32; y++ {
			for x := sx; x < sx+32; x++ {
				f.Y[y*width+x] = 220
			}
		}
		return f
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("threads-%d", tc.threads), func(t *testing.T) {
			e, err := NewVideoEncoder(width, height, 60)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := e.Close(); err != nil {
					t.Errorf("close encoder: %v", err)
				}
			})
			e.SetMaxThreads(tc.threads)
			if err := e.Prewarm(); err != nil {
				t.Fatal(err)
			}

			streamHash, reconHash := sha256.New(), sha256.New()
			for i := range frameCount {
				tu, _, err := e.Encode(makeFrame(i), false)
				if err != nil {
					t.Fatalf("frame %d encode: %v", i, err)
				}
				var length [8]byte
				binary.LittleEndian.PutUint64(length[:], uint64(len(tu)))
				_, _ = streamHash.Write(length[:])
				_, _ = streamHash.Write(tu)
				recon := e.Recon()
				_, _ = reconHash.Write(recon.Y)
				_, _ = reconHash.Write(recon.U)
				_, _ = reconHash.Write(recon.V)
			}
			if got := fmt.Sprintf("%x", streamHash.Sum(nil)); got != tc.stream {
				t.Errorf("bitstream SHA-256 = %s, want baseline %s", got, tc.stream)
			}
			if got := fmt.Sprintf("%x", reconHash.Sum(nil)); got != tc.recon {
				t.Errorf("reconstruction SHA-256 = %s, want baseline %s", got, tc.recon)
			}
		})
	}
}
