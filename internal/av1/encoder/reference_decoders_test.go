package encoder_test

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
)

const requireWebRTCReferenceDecodersEnv = "GOAV1_REQUIRE_WEBRTC_REFERENCE_DECODERS"

type referenceDecoderPaths struct {
	aomdec string
	dav1d  string
}

type lookPathFunc func(string) (string, error)

func resolveReferenceDecoders(lookPath lookPathFunc, required bool) (referenceDecoderPaths, string, error) {
	var paths referenceDecoderPaths
	var missing []string
	for _, name := range []string{"aomdec", "dav1d"} {
		path, err := lookPath(name)
		if err != nil || path == "" {
			missing = append(missing, name)
			continue
		}
		if name == "aomdec" {
			paths.aomdec = path
		} else {
			paths.dav1d = path
		}
	}
	if required && len(missing) != 0 {
		return paths, "", fmt.Errorf("%s=1 requires both aomdec and dav1d on PATH; missing: %s", requireWebRTCReferenceDecodersEnv, strings.Join(missing, ", "))
	}
	if !required && paths.aomdec == "" {
		return paths, "aomdec not on PATH; reference decoder checks are optional outside strict WebRTC gates", nil
	}
	return paths, "", nil
}

func referenceDecodersForTest(t *testing.T) referenceDecoderPaths {
	t.Helper()
	paths, skip, err := resolveReferenceDecoders(exec.LookPath, os.Getenv(requireWebRTCReferenceDecodersEnv) == "1")
	if err != nil {
		t.Fatal(err)
	}
	if skip != "" {
		t.Skip(skip)
	}
	t.Logf("reference decoder binaries: aomdec=%s dav1d=%s", paths.aomdec, paths.dav1d)
	return paths
}

func TestResolveReferenceDecoders(t *testing.T) {
	tests := []struct {
		name             string
		available        map[string]string
		wantPaths        referenceDecoderPaths
		wantMissing      []string
		wantOptionalSkip bool
	}{
		{
			name: "both available",
			available: map[string]string{
				"aomdec": "/tools/aomdec",
				"dav1d":  "/tools/dav1d",
			},
			wantPaths: referenceDecoderPaths{aomdec: "/tools/aomdec", dav1d: "/tools/dav1d"},
		},
		{
			name:        "dav1d missing",
			available:   map[string]string{"aomdec": "/tools/aomdec"},
			wantPaths:   referenceDecoderPaths{aomdec: "/tools/aomdec"},
			wantMissing: []string{"dav1d"},
		},
		{
			name:             "aomdec missing but dav1d available",
			available:        map[string]string{"dav1d": "/tools/dav1d"},
			wantPaths:        referenceDecoderPaths{dav1d: "/tools/dav1d"},
			wantMissing:      []string{"aomdec"},
			wantOptionalSkip: true,
		},
		{
			name:             "both missing",
			available:        map[string]string{},
			wantMissing:      []string{"aomdec", "dav1d"},
			wantOptionalSkip: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lookup := func(name string) (string, error) {
				if path, ok := tt.available[name]; ok {
					return path, nil
				}
				return "", exec.ErrNotFound
			}

			for _, required := range []bool{false, true} {
				paths, skip, err := resolveReferenceDecoders(lookup, required)
				if paths != tt.wantPaths {
					t.Fatalf("resolveReferenceDecoders() paths = %#v, want %#v", paths, tt.wantPaths)
				}
				if !required && (skip != "") != tt.wantOptionalSkip {
					t.Fatalf("optional skip reason = %q, want skip=%v", skip, tt.wantOptionalSkip)
				}
				if required && len(tt.wantMissing) == 0 && err != nil {
					t.Fatalf("required decoder resolution failed: %v", err)
				}
				if required && len(tt.wantMissing) != 0 {
					if err == nil {
						t.Fatal("required decoder resolution succeeded with missing tools")
					}
					for _, name := range tt.wantMissing {
						if !strings.Contains(err.Error(), name) {
							t.Errorf("required decoder error %q does not identify missing %s", err, name)
						}
					}
				}
			}
		})
	}
}
