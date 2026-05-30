package testvector

import (
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// OracleKind describes the external oracle data associated with a remote
// vector. Libaom AV1 vectors use per-frame MD5 files.
type OracleKind uint8

const (
	OracleFrameMD5 OracleKind = iota + 1
)

// SuiteLevel lets callers select a small smoke set, a feature-relevant subset,
// an extended cohort that probes latent decoder gaps (multi-quantizer 10-bit,
// larger sizes, SVC permutations), or the whole committed remote manifest.
// SuiteLevelExtended is strictly opt-in: it is never part of the default fast
// gate, but every extended vector is also tagged SuiteLevelFull so
// testvectors-full keeps downloading the complete manifest.
type SuiteLevel uint8

const (
	SuiteLevelFast SuiteLevel = 1 << iota
	SuiteLevelRelevant
	SuiteLevelFull
	SuiteLevelExtended
)

// VectorLabel is a bitset used for relevant-suite filtering.
type VectorLabel uint64

const (
	VectorLabelQuantizer VectorLabel = 1 << iota
	VectorLabelSize
	VectorLabelIntra
	VectorLabelIntrabc
	VectorLabelCDF
	VectorLabelMotion
	VectorLabelMFMV
	VectorLabelSVC
	VectorLabelFilmGrain
	VectorLabelMonochrome
	VectorLabelHighBitDepth
	// VectorLabelMultiTile marks streams that exercise the multi-tile
	// decode path (TileInfo.Cols > 1 or TileInfo.Rows > 1). The libaom
	// v3.14.0 published AV1 test suite does not include a dedicated
	// multi-tile bitstream (e.g. "av1-1-b8-XX-tiles.ivf"); the only
	// committed multi-tile vectors in the manifest are the SVC ones:
	// L1T2 has Cols=3 on every frame, L2T1 / L2T2 emit Cols=3 base
	// layers and Cols=4 enhancement layers. Selecting by this label
	// via SelectRemote(0, VectorLabelMultiTile, nil) returns the
	// multi-tile probe cohort.
	VectorLabelMultiTile
)

// RemoteFile describes one checksum-pinned test-data object. Name is a single
// path component under the provider's base URL and local cache directory.
type RemoteFile struct {
	Name string
	SHA1 string
}

// RemoteVector describes one external vector stream and its oracle file. The
// bytes are intentionally not stored in git; callers download into an ignored
// cache after validating this metadata.
type RemoteVector struct {
	Tag    Tag
	Kind   Kind
	Name   string
	Stream RemoteFile
	MD5    RemoteFile
	Oracle OracleKind
	Levels SuiteLevel
	Labels VectorLabel
}

// RemoteManifest is committed metadata for checksum-pinned external vectors.
type RemoteManifest struct {
	Name    string
	Source  string
	BaseURL string
	Vectors []RemoteVector
}

func (f RemoteFile) Validate() error {
	if _, err := f.SHA1Bytes(); err != nil {
		return err
	}
	if f.Name == "" || filepath.Base(f.Name) != f.Name ||
		strings.ContainsAny(f.Name, `/\`) || strings.Contains(f.Name, "..") {
		return ErrInvalidRemote
	}
	return nil
}

func (f RemoteFile) SHA1Bytes() ([20]byte, error) {
	if len(f.SHA1) != 40 {
		return [20]byte{}, ErrInvalidRemote
	}
	var dst [20]byte
	if _, err := hex.Decode(dst[:], []byte(f.SHA1)); err != nil {
		return [20]byte{}, ErrInvalidRemote
	}
	return dst, nil
}

func (m RemoteManifest) Validate() error {
	if m.Name == "" || m.BaseURL == "" || len(m.Vectors) == 0 {
		return ErrInvalidRemote
	}
	for i, vector := range m.Vectors {
		if vector.Tag == 0 || vector.Kind == 0 || vector.Name == "" ||
			vector.Oracle == 0 || vector.Levels == 0 {
			return ErrInvalidRemote
		}
		if err := vector.Stream.Validate(); err != nil {
			return err
		}
		if err := vector.MD5.Validate(); err != nil {
			return err
		}
		for j := range i {
			other := m.Vectors[j]
			if other.Tag == vector.Tag ||
				other.Stream.Name == vector.Stream.Name ||
				other.MD5.Name == vector.MD5.Name {
				return ErrDuplicateTag
			}
		}
	}
	return nil
}

func (m RemoteManifest) FindRemote(tag Tag) (RemoteVector, bool) {
	for _, vector := range m.Vectors {
		if vector.Tag == tag {
			return vector, true
		}
	}
	return RemoteVector{}, false
}

func (m RemoteManifest) SelectRemote(level SuiteLevel, labels VectorLabel, dst []RemoteVector) []RemoteVector {
	for _, vector := range m.Vectors {
		if level != 0 && vector.Levels&level == 0 {
			continue
		}
		if labels != 0 && vector.Labels&labels == 0 {
			continue
		}
		dst = append(dst, vector)
	}
	return dst
}

// LoadRemoteSuite downloads the selected remote vectors into cacheDir and
// returns a byte-backed test suite plus parsed oracle digests. The downloaded
// files stay in the caller's ignored cache; only checksum-pinned metadata is
// expected to live in git.
func LoadRemoteSuite(ctx context.Context, manifest RemoteManifest, cacheDir string, level SuiteLevel, labels VectorLabel) (Suite, error) {
	client := &http.Client{Timeout: 30 * time.Second}
	return LoadRemoteSuiteWithClient(ctx, client, manifest, cacheDir, level, labels)
}

func LoadRemoteSuiteWithClient(ctx context.Context, client *http.Client, manifest RemoteManifest, cacheDir string, level SuiteLevel, labels VectorLabel) (Suite, error) {
	if client == nil || cacheDir == "" {
		return Suite{}, ErrInvalidRemote
	}
	if err := manifest.Validate(); err != nil {
		return Suite{}, err
	}
	selected := manifest.SelectRemote(level, labels, nil)
	if len(selected) == 0 {
		return Suite{}, ErrMissingVector
	}

	vectors := make([]Vector, 0, len(selected))
	var digests []FrameDigest
	for _, remote := range selected {
		streamPath, err := EnsureRemoteFileWithClient(ctx, client, cacheDir, manifest.BaseURL, remote.Stream)
		if err != nil {
			return Suite{}, err
		}
		streamData, err := os.ReadFile(streamPath)
		if err != nil {
			return Suite{}, err
		}
		vectors = append(vectors, Vector{
			Tag:   remote.Tag,
			Kind:  remote.Kind,
			Name:  remote.Name,
			Input: streamData,
		})

		switch remote.Oracle {
		case OracleFrameMD5:
			md5Path, err := EnsureRemoteFileWithClient(ctx, client, cacheDir, manifest.BaseURL, remote.MD5)
			if err != nil {
				return Suite{}, err
			}
			md5Data, err := os.ReadFile(md5Path)
			if err != nil {
				return Suite{}, err
			}
			parsed, err := ParseLibaomMD5Digests(remote.Tag, md5Data)
			if err != nil {
				return Suite{}, err
			}
			digests = append(digests, parsed...)
		default:
			return Suite{}, ErrInvalidRemote
		}
	}

	return Suite{
		Name: manifest.Name,
		Manifest: Manifest{
			Vectors: vectors,
			Digests: digests,
		},
	}, nil
}

// EnsureRemoteFile returns a checksum-verified cache path for file, downloading
// it from baseURL only when the cached copy is absent or corrupted.
func EnsureRemoteFile(ctx context.Context, cacheDir string, baseURL string, file RemoteFile) (string, error) {
	client := &http.Client{Timeout: 30 * time.Second}
	return EnsureRemoteFileWithClient(ctx, client, cacheDir, baseURL, file)
}

func EnsureRemoteFileWithClient(ctx context.Context, client *http.Client, cacheDir string, baseURL string, file RemoteFile) (string, error) {
	if client == nil || cacheDir == "" || baseURL == "" {
		return "", ErrInvalidRemote
	}
	want, err := file.SHA1Bytes()
	if err != nil {
		return "", err
	}
	if err := file.Validate(); err != nil {
		return "", err
	}
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(cacheDir, file.Name)
	if ok, err := fileMatchesRemoteSHA1(path, want); err != nil {
		return "", err
	} else if ok {
		return path, nil
	}

	tmp, err := os.CreateTemp(cacheDir, "."+file.Name+".tmp-*")
	if err != nil {
		return "", err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := downloadRemoteFile(ctx, client, tmp, strings.TrimRight(baseURL, "/")+"/"+file.Name); err != nil {
		return "", err
	}
	if ok, err := fileMatchesRemoteSHA1(tmpName, want); err != nil {
		return "", err
	} else if !ok {
		return "", ErrChecksum
	}
	if err := os.Rename(tmpName, path); err != nil {
		if removeErr := os.Remove(path); removeErr != nil && !os.IsNotExist(removeErr) {
			return "", err
		}
		if err := os.Rename(tmpName, path); err != nil {
			return "", err
		}
	}
	return path, nil
}

func downloadRemoteFile(ctx context.Context, client *http.Client, out *os.File, url string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		_ = out.Close()
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		_ = out.Close()
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		_ = out.Close()
		return fmt.Errorf("download %s: %s", url, resp.Status)
	}
	_, copyErr := io.Copy(out, resp.Body)
	closeErr := out.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}

func fileMatchesRemoteSHA1(path string, want [20]byte) (bool, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
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
	return bytes.Equal(h.Sum(nil), want[:]), nil
}
