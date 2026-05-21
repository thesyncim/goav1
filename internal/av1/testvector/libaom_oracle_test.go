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
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
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

func TestLibaomQuantizer00FrameWorkDryRun(t *testing.T) {
	ivfData := readLibaomRemoteFile(t, libaomQuantizer00IVF)
	md5Data := readLibaomRemoteFile(t, libaomQuantizer00MD5)
	digests := parseLibaomMD5Digests(t, TagDecoderLibaomQuantizer00, md5Data)

	it, err := ivf.NewIterator(ivfData)
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	var stream decoder.Stream
	var refs decoder.SurfaceReferences
	var state decoder.FrameWorkState
	var pool frame.Pool
	var poolFormat frame.Format
	var havePool bool
	var backing []byte
	var frameSlots []frame.Frame
	var free []int
	var used []bool

	var events [16]decoder.Event
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [parser.MaxTiles]parser.TileSpan
	var jobs [parser.MaxTiles]tile.Job
	var batches [parser.MaxTiles]threading.Batch
	var releases [parser.RefFrames]int

	frameCount := 0
	completed := 0
	tileJobs := 0
	retainedContexts := 0
	for {
		ivfFrame, ok, err := it.Next()
		if err != nil {
			t.Fatalf("ivf frame %d: %v", frameCount, err)
		}
		if !ok {
			break
		}
		count, err := stream.PushLowOverhead(ivfFrame.Payload, events[:])
		if err != nil {
			t.Fatalf("frame %d PushLowOverhead: %v", ivfFrame.Index, err)
		}
		for i := 0; i < count; i++ {
			event := events[i]
			if !eventRunsFrameWork(event) {
				continue
			}
			if !havePool {
				pool, poolFormat, backing, frameSlots, free, used = bindLibaomVectorFramePool(t, event, 8)
				havePool = true
			} else {
				gotFormat := frameFormatFromEvent(event)
				if gotFormat != poolFormat {
					t.Fatalf("frame %d format=%+v want %+v", ivfFrame.Index, gotFormat, poolFormat)
				}
			}

			var postMD5 MD5
			postRan := false
			result, err := state.RunEventWithContextAndPostFilter(&refs, &pool, event.SequenceHeader, event, 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, func(ctx decoder.FrameWorkBatch) error {
				for j := 0; j < len(ctx.Jobs); j++ {
					var decodeState tile.DecodeState
					if err := ctx.JobDecodeState(j, &decodeState); err != nil {
						return err
					}
					if _, err := ctx.JobRegion(j); err != nil {
						return err
					}
					if _, err := ctx.JobOutputPlane(j, threading.FrameWorkPlaneY); err != nil {
						return err
					}
					if _, err := ctx.LoopRestorationPlan(false); err != nil {
						return err
					}
					tileJobs++
					if decodeState.RetainFrameContext {
						retainedContexts++
					}
				}
				return nil
			}, func(ctx decoder.FrameWorkPostFilterContext) error {
				digestIndex := int(ivfFrame.Index)
				if digestIndex >= len(digests) || digests[digestIndex].FrameIndex != ivfFrame.Index {
					t.Fatalf("frame %d missing official digest", ivfFrame.Index)
				}
				got, err := FrameMD5(*ctx.Output)
				if err != nil {
					return err
				}
				postMD5 = got
				postRan = true
				return nil
			})
			if err != nil {
				t.Fatalf("frame %d RunEventWithContextAndPostFilter: %v", ivfFrame.Index, err)
			}
			if result.Run.CompletedFrame {
				if !postRan {
					t.Fatalf("frame %d completed without postfilter", ivfFrame.Index)
				}
				digestIndex := int(ivfFrame.Index)
				if digestIndex >= len(digests) {
					t.Fatalf("frame %d missing official digest", ivfFrame.Index)
				}
				if postMD5 == digests[digestIndex].MD5 {
					t.Fatalf("frame %d unexpectedly matched official md5 before reconstruction is wired", ivfFrame.Index)
				}
				completed++
			}
		}
		frameCount++
	}
	if frameCount != len(digests) || completed != len(digests) {
		t.Fatalf("frames=%d completed=%d want %d", frameCount, completed, len(digests))
	}
	if tileJobs == 0 {
		t.Fatal("no tile jobs ran")
	}
	if retainedContexts == 0 {
		t.Fatal("no context-update tile was retained")
	}
	runtime.KeepAlive(backing)
	runtime.KeepAlive(frameSlots)
	runtime.KeepAlive(free)
	runtime.KeepAlive(used)
}

func eventCompletesDecodedFrame(event decoder.Event) bool {
	return (event.Kind == decoder.EventFrame || event.Kind == decoder.EventTileGroup) && event.TileGroup.Final
}

func eventRunsFrameWork(event decoder.Event) bool {
	switch event.Kind {
	case decoder.EventFrameHeader, decoder.EventFrame, decoder.EventTileGroup, decoder.EventExistingFrame:
		return true
	default:
		return false
	}
}

func bindLibaomVectorFramePool(t *testing.T, event decoder.Event, count int) (frame.Pool, frame.Format, []byte, []frame.Frame, []int, []bool) {
	t.Helper()
	format := frameFormatFromEvent(event)
	layout, err := frame.RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*count)
	frames := make([]frame.Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	pool, err := frame.BindPool(backing, format, frames, free, used)
	if err != nil {
		t.Fatal(err)
	}
	return pool, format, backing, frames, free, used
}

func frameFormatFromEvent(event decoder.Event) frame.Format {
	return frame.Format{
		Width:        int(event.FrameSize.CodedWidth),
		Height:       int(event.FrameSize.Height),
		BitDepth:     event.SequenceHeader.ColorConfig.BitDepth,
		MonoChrome:   event.SequenceHeader.ColorConfig.MonoChrome,
		SubsamplingX: event.SequenceHeader.ColorConfig.SubsamplingX,
		SubsamplingY: event.SequenceHeader.ColorConfig.SubsamplingY,
		Align:        32,
	}
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
