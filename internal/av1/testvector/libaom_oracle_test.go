//go:build goav1_oracle

package testvector

import (
	"bytes"
	"crypto/sha1"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/thesyncim/goav1/internal/av1/decoder"
	"github.com/thesyncim/goav1/internal/av1/ivf"
)

const libaomTestDataURL = "https://storage.googleapis.com/aom-test-data"

type libaomRemoteFile struct {
	name string
	sha1 [20]byte
}

var libaomQuantizer00IVF = libaomRemoteFile{
	name: "av1-1-b8-00-quantizer-00.ivf",
	sha1: [20]byte{
		0xc2, 0xe1, 0xec, 0x99, 0x36, 0xb9, 0x52, 0x54, 0x18, 0x7a,
		0x35, 0x9e, 0x94, 0xaa, 0x32, 0xa9, 0xf3, 0xda, 0xd1, 0xb7,
	},
}

var libaomQuantizer00MD5 = libaomRemoteFile{
	name: "av1-1-b8-00-quantizer-00.ivf.md5",
	sha1: [20]byte{
		0x26, 0xcd, 0x2a, 0x03, 0x21, 0xd0, 0x1d, 0x9d, 0xb5, 0xf6,
		0xda, 0xce, 0x8b, 0x43, 0xa4, 0x0c, 0xd5, 0xb9, 0xd5, 0x8d,
	},
}

func TestLibaomQuantizer00OfficialMD5Manifest(t *testing.T) {
	md5Data := readLibaomRemoteFile(t, libaomQuantizer00MD5)
	digests := parseLibaomMD5Digests(t, TagDecoderLibaomQuantizer00, md5Data)
	if err := (Manifest{Digests: digests}).Validate(); err != nil {
		t.Fatal(err)
	}
	if len(digests) != 2 {
		t.Fatalf("digest count=%d want 2", len(digests))
	}

	oracle := NewOracle(Manifest{Digests: digests})
	for _, digest := range digests {
		if err := oracle.CheckMD5(digest.Tag, digest.FrameIndex, digest.MD5); err != nil {
			t.Fatalf("official md5 oracle frame=%d err=%v", digest.FrameIndex, err)
		}
	}
}

func TestLibaomQuantizer00StreamEvents(t *testing.T) {
	ivfData := readLibaomRemoteFile(t, libaomQuantizer00IVF)
	md5Data := readLibaomRemoteFile(t, libaomQuantizer00MD5)
	digests := parseLibaomMD5Digests(t, TagDecoderLibaomQuantizer00, md5Data)
	if err := (Manifest{Digests: digests}).Validate(); err != nil {
		t.Fatal(err)
	}

	it, err := ivf.NewIterator(ivfData)
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}

	var stream decoder.Stream
	var events [16]decoder.Event
	frameCount := 0
	oracle := NewOracle(Manifest{Digests: digests})
	for {
		frame, ok, err := it.Next()
		if err != nil {
			t.Fatalf("ivf frame %d: %v", frameCount, err)
		}
		if !ok {
			break
		}

		count, err := stream.PushLowOverhead(frame.Payload, events[:])
		if err != nil {
			t.Fatalf("frame %d PushLowOverhead: %v", frame.Index, err)
		}
		if count == 0 {
			t.Fatalf("frame %d produced no decoder events", frame.Index)
		}

		final := false
		for i := 0; i < count; i++ {
			event := events[i]
			if !eventCompletesDecodedFrame(event) {
				continue
			}
			digestIndex := int(frame.Index)
			if digestIndex >= len(digests) {
				t.Fatalf("frame %d has no official md5 digest", frame.Index)
			}
			if digests[digestIndex].FrameIndex != frame.Index {
				t.Fatalf("frame digest index=%d want %d", digests[digestIndex].FrameIndex, frame.Index)
			}
			final = true
			if event.FrameSize.CodedWidth != 352 || event.FrameSize.Height != 288 {
				t.Fatalf("frame %d size=%dx%d want 352x288", frame.Index, event.FrameSize.CodedWidth, event.FrameSize.Height)
			}
			if event.TileGroup.TileCount == 0 || event.TileGroup.DataSize == 0 {
				t.Fatalf("frame %d tile group=%+v", frame.Index, event.TileGroup)
			}
			if err := oracle.CheckMD5(TagDecoderLibaomQuantizer00, frame.Index, digests[digestIndex].MD5); err != nil {
				t.Fatalf("frame %d official md5 err=%v", frame.Index, err)
			}
		}
		if !final {
			t.Fatalf("frame %d did not produce a final frame event", frame.Index)
		}
		frameCount++
	}
	if frameCount != len(digests) {
		t.Fatalf("frame count=%d want %d", frameCount, len(digests))
	}
}

func eventCompletesDecodedFrame(event decoder.Event) bool {
	return (event.Kind == decoder.EventFrame || event.Kind == decoder.EventTileGroup) && event.TileGroup.Final
}

func parseLibaomMD5Digests(t *testing.T, tag Tag, src []byte) []FrameDigest {
	t.Helper()
	lines := bytes.Split(bytes.TrimSpace(src), []byte{'\n'})
	digests := make([]FrameDigest, 0, len(lines))
	for i, line := range lines {
		fields := bytes.Fields(line)
		if len(fields) == 0 {
			continue
		}
		md5, err := ParseMD5Hex(fields[0])
		if err != nil {
			t.Fatalf("md5 line %d: %v", i, err)
		}
		digests = append(digests, FrameDigest{
			Tag:        tag,
			FrameIndex: uint32(len(digests)),
			MD5:        md5,
		})
	}
	if len(digests) == 0 {
		t.Fatal("empty libaom md5 manifest")
	}
	return digests
}

func readLibaomRemoteFile(t *testing.T, file libaomRemoteFile) []byte {
	t.Helper()
	path := ensureLibaomRemoteFile(t, file)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := sha1.Sum(data); got != file.sha1 {
		t.Fatalf("%s sha1=%x want %x", file.name, got, file.sha1)
	}
	return data
}

func ensureLibaomRemoteFile(t *testing.T, file libaomRemoteFile) string {
	t.Helper()
	dir := libaomCacheDir(t)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, file.name)
	if ok, err := fileMatchesSHA1(path, file.sha1); err != nil {
		t.Fatal(err)
	} else if ok {
		return path
	}

	tmp := path + ".tmp"
	if err := downloadLibaomRemoteFile(tmp, file); err != nil {
		t.Fatal(err)
	}
	if ok, err := fileMatchesSHA1(tmp, file.sha1); err != nil {
		_ = os.Remove(tmp)
		t.Fatal(err)
	} else if !ok {
		_ = os.Remove(tmp)
		t.Fatalf("%s downloaded sha1 mismatch", file.name)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		t.Fatal(err)
	}
	return path
}

func downloadLibaomRemoteFile(dst string, file libaomRemoteFile) error {
	url := libaomTestDataURL + "/" + file.name
	client := http.Client{Timeout: 30 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: %s", file.name, resp.Status)
	}

	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(out, resp.Body)
	closeErr := out.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}

func fileMatchesSHA1(path string, want [20]byte) (bool, error) {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	defer f.Close()

	h := sha1.New()
	if _, err := io.Copy(h, f); err != nil {
		return false, err
	}
	sum := h.Sum(nil)
	return bytes.Equal(sum, want[:]), nil
}

func libaomCacheDir(t *testing.T) string {
	t.Helper()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(filename), "..", "testdata", "libaom"))
}
