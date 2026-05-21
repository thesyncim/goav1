package testvector

import (
	"context"
	"crypto/sha1"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestRemoteManifestValidateFindAndSelect(t *testing.T) {
	content := []byte("stream")
	md5 := []byte("md5")
	manifest := RemoteManifest{
		Name:    "local",
		BaseURL: "https://example.invalid",
		Vectors: []RemoteVector{{
			Tag:    TagDecoderLibaomQuantizer00,
			Kind:   KindDecoder,
			Name:   "quantizer",
			Stream: RemoteFile{Name: "stream.ivf", SHA1: sha1Hex(content)},
			MD5:    RemoteFile{Name: "stream.ivf.md5", SHA1: sha1Hex(md5)},
			Oracle: OracleFrameMD5,
			Levels: SuiteLevelFast | SuiteLevelFull,
			Labels: VectorLabelQuantizer,
		}},
	}
	if err := manifest.Validate(); err != nil {
		t.Fatal(err)
	}
	vector, ok := manifest.FindRemote(TagDecoderLibaomQuantizer00)
	if !ok {
		t.Fatal("missing remote vector")
	}
	if vector.Stream.Name != "stream.ivf" {
		t.Fatalf("vector=%+v", vector)
	}
	selected := manifest.SelectRemote(SuiteLevelFast, VectorLabelQuantizer, nil)
	if len(selected) != 1 || selected[0].Tag != TagDecoderLibaomQuantizer00 {
		t.Fatalf("selected=%+v", selected)
	}
	if selected := manifest.SelectRemote(SuiteLevelRelevant, VectorLabelMotion, nil); len(selected) != 0 {
		t.Fatalf("unexpected selection=%+v", selected)
	}
}

func TestRemoteManifestRejectsInvalidMetadata(t *testing.T) {
	if err := (RemoteManifest{}).Validate(); !errors.Is(err, ErrInvalidRemote) {
		t.Fatalf("empty manifest err=%v want %v", err, ErrInvalidRemote)
	}
	if err := (RemoteFile{Name: "../bad.ivf", SHA1: sha1Hex([]byte("x"))}).Validate(); !errors.Is(err, ErrInvalidRemote) {
		t.Fatalf("bad path err=%v want %v", err, ErrInvalidRemote)
	}
	if err := (RemoteFile{Name: "bad.ivf", SHA1: "bad"}).Validate(); !errors.Is(err, ErrInvalidRemote) {
		t.Fatalf("bad sha1 err=%v want %v", err, ErrInvalidRemote)
	}

	file := RemoteFile{Name: "same.ivf", SHA1: sha1Hex([]byte("x"))}
	dup := RemoteManifest{
		Name:    "dup",
		BaseURL: "https://example.invalid",
		Vectors: []RemoteVector{
			{Tag: TagDecoderLibaomQuantizer00, Kind: KindDecoder, Name: "one", Stream: file, MD5: RemoteFile{Name: "one.md5", SHA1: sha1Hex([]byte("1"))}, Oracle: OracleFrameMD5, Levels: SuiteLevelFast},
			{Tag: TagDecoderLibaomQuantizer00, Kind: KindDecoder, Name: "two", Stream: RemoteFile{Name: "two.ivf", SHA1: sha1Hex([]byte("2"))}, MD5: RemoteFile{Name: "two.md5", SHA1: sha1Hex([]byte("3"))}, Oracle: OracleFrameMD5, Levels: SuiteLevelFast},
		},
	}
	if err := dup.Validate(); !errors.Is(err, ErrDuplicateTag) {
		t.Fatalf("duplicate err=%v want %v", err, ErrDuplicateTag)
	}
}

func TestEnsureRemoteFileDownloadsCachesAndRepairs(t *testing.T) {
	payload := []byte("remote payload")
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Path != "/vector.ivf" {
			t.Fatalf("path=%q want /vector.ivf", r.URL.Path)
		}
		_, _ = w.Write(payload)
	}))
	defer server.Close()

	dir := t.TempDir()
	file := RemoteFile{Name: "vector.ivf", SHA1: sha1Hex(payload)}
	path, err := EnsureRemoteFile(context.Background(), dir, server.URL, file)
	if err != nil {
		t.Fatal(err)
	}
	if path != filepath.Join(dir, file.Name) {
		t.Fatalf("path=%q", path)
	}
	if requests != 1 {
		t.Fatalf("requests=%d want 1", requests)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(payload) {
		t.Fatalf("payload=%q want %q", got, payload)
	}

	path, err = EnsureRemoteFile(context.Background(), dir, server.URL, file)
	if err != nil {
		t.Fatal(err)
	}
	if requests != 1 {
		t.Fatalf("cached requests=%d want 1", requests)
	}
	if err := os.WriteFile(path, []byte("corrupt"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := EnsureRemoteFile(context.Background(), dir, server.URL, file); err != nil {
		t.Fatal(err)
	}
	if requests != 2 {
		t.Fatalf("repair requests=%d want 2", requests)
	}
}

func TestEnsureRemoteFileRejectsChecksumMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("wrong"))
	}))
	defer server.Close()

	file := RemoteFile{Name: "vector.ivf", SHA1: sha1Hex([]byte("right"))}
	if _, err := EnsureRemoteFile(context.Background(), t.TempDir(), server.URL, file); !errors.Is(err, ErrChecksum) {
		t.Fatalf("err=%v want %v", err, ErrChecksum)
	}
}

func sha1Hex(data []byte) string {
	sum := sha1.Sum(data)
	return fmt.Sprintf("%x", sum)
}
