package main

import (
	"context"
	"errors"
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
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		raw, err := os.ReadFile(pidPath)
		if err == nil {
			value := strings.TrimSpace(string(raw))
			if value == "" {
				time.Sleep(5 * time.Millisecond)
				continue
			}
			pid, err := strconv.Atoi(value)
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

func killQualitybenchTestHelperChild(pidPath string) error {
	raw, err := os.ReadFile(pidPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("read helper child pid: %w", err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(raw)))
	if err != nil {
		return fmt.Errorf("invalid helper child pid %q: %w", raw, err)
	}
	if pid < 1 {
		return fmt.Errorf("invalid helper child pid %q: pid must be positive", raw)
	}
	process, err := os.FindProcess(pid)
	if err != nil {
		return fmt.Errorf("find helper child %d: %w", pid, err)
	}
	if err := process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
		return fmt.Errorf("kill helper child %d: %w", pid, err)
	}
	// On Windows, the executable remains locked until its process handle is
	// waited and released. The child is not our direct child there, but Wait
	// still observes its exit through the process handle returned by FindProcess.
	if runtime.GOOS == "windows" {
		if _, err := process.Wait(); err != nil {
			return fmt.Errorf("wait for helper child %d to exit: %w", pid, err)
		}
	}
	if err := os.Remove(pidPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove helper child pid file: %w", err)
	}
	return nil
}

func TestCommandWaitDelayWithInheritedPipes(t *testing.T) {
	bin := qualitybenchTestHelper(t)
	pidPath := filepath.Join(t.TempDir(), "wait-delay-child.pid")
	ctx, cancel := context.WithCancel(context.Background())
	cmd := commandContextWithWaitDelay(ctx, bin, qualitybenchHelperArgs("hold-pipes", pidPath)...)
	if cmd.WaitDelay != commandWaitDelay {
		cancel()
		t.Fatalf("command wait delay=%s, want %s", cmd.WaitDelay, commandWaitDelay)
	}
	cmd.Env = externalCommandEnv()
	var output boundedCommandOutput
	cmd.Stdout = &output
	cmd.Stderr = &output
	if err := cmd.Start(); err != nil {
		cancel()
		t.Fatalf("start command with inherited-pipe child: %v", err)
	}
	waited := false
	t.Cleanup(func() {
		cancel()
		if !waited {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
		if err := killQualitybenchTestHelperChild(pidPath); err != nil {
			t.Errorf("clean up inherited-pipe helper child: %v", err)
		}
	})

	qualitybenchTestHelperPID(t, pidPath)
	cancelledAt := time.Now()
	cancel()
	err := cmd.Wait()
	waited = true
	if err == nil {
		t.Fatal("cancelled command returned without an error")
	}
	if elapsed := time.Since(cancelledAt); elapsed > commandWaitDelay+time.Second {
		t.Fatalf("command wait after cancellation=%s, want at most %s plus scheduling margin",
			elapsed, commandWaitDelay)
	}
	if err := killQualitybenchTestHelperChild(pidPath); err != nil {
		t.Fatalf("kill and reap inherited-pipe helper child: %v", err)
	}
}

func TestMain(m *testing.M) {
	code := m.Run()
	if qualitybenchHelperDir != "" {
		if err := os.RemoveAll(qualitybenchHelperDir); err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "remove qualitybench test helper: %v\n", err)
			if code == 0 {
				code = 1
			}
		}
	}
	os.Exit(code)
}
