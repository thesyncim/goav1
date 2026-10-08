package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

var (
	qualitybenchHelperOnce sync.Once
	qualitybenchHelperPath string
	qualitybenchHelperDir  string
	qualitybenchHelperErr  error
)

func qualitybenchTestHelper(t *testing.T) string {
	t.Helper()
	qualitybenchHelperOnce.Do(func() {
		_, thisFile, _, ok := runtime.Caller(0)
		if !ok {
			qualitybenchHelperErr = fmt.Errorf("locate qualitybench test helper source")
			return
		}
		packageDir := filepath.Dir(thisFile)
		qualitybenchHelperDir, qualitybenchHelperErr = os.MkdirTemp("", "goav1-qualitybench-helper-")
		if qualitybenchHelperErr != nil {
			return
		}
		exeName := "qualitybench-helper"
		goName := "go"
		if runtime.GOOS == "windows" {
			exeName += ".exe"
			goName += ".exe"
		}
		qualitybenchHelperPath = filepath.Join(qualitybenchHelperDir, exeName)
		goBin := filepath.Join(runtime.GOROOT(), "bin", goName)
		source := filepath.Join(packageDir, "testdata", "qualitybench_helper.go")
		cmd := exec.Command(goBin, "build", "-o", qualitybenchHelperPath, source)
		cmd.Dir = packageDir
		out, err := cmd.CombinedOutput()
		if err != nil {
			qualitybenchHelperErr = fmt.Errorf("build qualitybench test helper: %w\n%s", err, strings.TrimSpace(string(out)))
		}
	})
	if qualitybenchHelperErr != nil {
		t.Fatalf("qualitybench test helper unavailable: %v", qualitybenchHelperErr)
	}
	return qualitybenchHelperPath
}

func qualitybenchHelperArgs(mode string, args ...string) []string {
	return append([]string{"--qualitybench-helper=" + mode}, args...)
}

func qualitybenchExecutableName(name string) string {
	if runtime.GOOS == "windows" && filepath.Ext(name) == "" {
		return name + ".exe"
	}
	return name
}

func qualitybenchTestHelperPID(t *testing.T, pidPath string) int {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		raw, err := os.ReadFile(pidPath)
		if err == nil {
			pid, err := strconv.Atoi(strings.TrimSpace(string(raw)))
			if err != nil || pid < 1 {
				t.Fatalf("invalid helper child pid %q: %v", raw, err)
			}
			return pid
		}
		if !os.IsNotExist(err) {
			t.Fatalf("read helper child pid: %v", err)
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("helper did not record child pid in %q", pidPath)
	return 0
}

func killQualitybenchTestHelperChild(pidPath string) {
	raw, err := os.ReadFile(pidPath)
	if err != nil {
		return
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(raw)))
	if err != nil || pid < 1 {
		return
	}
	process, err := os.FindProcess(pid)
	if err == nil {
		_ = process.Kill()
	}
}

func waitForQualitybenchCommand(t *testing.T, pidPath string, run func()) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		run()
		close(done)
	}()
	select {
	case <-done:
		return
	case <-time.After(1500 * time.Millisecond):
		killQualitybenchTestHelperChild(pidPath)
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatalf("timed command did not return after killing inherited-pipe child")
		}
		t.Fatalf("timed command did not return promptly while a child held stdout and stderr")
	}
}

func TestMain(m *testing.M) {
	code := m.Run()
	if qualitybenchHelperDir != "" {
		if err := os.RemoveAll(qualitybenchHelperDir); err != nil && code == 0 {
			_, _ = fmt.Fprintf(os.Stderr, "remove qualitybench test helper: %v\n", err)
			code = 1
		}
	}
	os.Exit(code)
}
