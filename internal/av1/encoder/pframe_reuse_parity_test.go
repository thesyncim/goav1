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
		{1, "fe30b65971282e7aa522074e5585a0860846af961a231e5c79703b9a49fd6521", "6f69305d99055aa85913c6bce1f26328282f79b5799ca63c19a4d4f39f0f19f8"},
		{4, "9b9ba2cf457e115d1c7b4e69efb3f70290ede3b32038eff97d7bedc735d938b3", "15d2208e01323d5810d12f11f5187439bfb269a832e33689e8940b2cfb42e4c4"},
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
