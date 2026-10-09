// This small executable is compiled by qualitybench tests that need a portable
// stand-in for an external command. Keeping it in testdata avoids adding it to
// the normal package build.
package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

func main() {
	if len(os.Args) > 1 && strings.HasPrefix(os.Args[1], "--qualitybench-helper=") {
		mode := strings.TrimPrefix(os.Args[1], "--qualitybench-helper=")
		runHelperMode(mode, os.Args[2:])
		return
	}
	runFFmpeg(os.Args[1:])
}

func runHelperMode(mode string, args []string) {
	switch mode {
	case "fail-once":
		if len(args) != 1 {
			fail("fail-once requires a counter path")
		}
		path := args[0]
		count := 0
		if raw, err := os.ReadFile(path); err == nil {
			count, _ = strconv.Atoi(strings.TrimSpace(string(raw)))
		}
		count++
		if err := os.WriteFile(path, []byte(strconv.Itoa(count)), 0o600); err != nil {
			fail("write counter: %v", err)
		}
		fmt.Printf("stdout-run-%d\n", count)
		fmt.Fprintf(os.Stderr, "stderr-run-%d\n", count)
		os.Exit(7)
	case "env":
		if len(args) != 1 {
			fail("env requires an output path")
		}
		f, err := os.Create(args[0])
		if err != nil {
			fail("create env output: %v", err)
		}
		for _, entry := range []struct{ label, key string }{
			{"omp", "OMP_NUM_THREADS"},
			{"dyld", "DYLD_INSERT_LIBRARIES"},
			{"lc", "LC_ALL"},
			{"tz", "TZ"},
			{"path", "PATH"},
		} {
			if _, err := fmt.Fprintf(f, "%s=%s\n", entry.label, os.Getenv(entry.key)); err != nil {
				_ = f.Close()
				fail("write env output: %v", err)
			}
		}
		if err := f.Close(); err != nil {
			fail("close env output: %v", err)
		}
	case "hold-pipes":
		if len(args) != 1 {
			fail("hold-pipes requires a pid path")
		}
		child := exec.Command(os.Args[0], "--qualitybench-helper=pipe-child")
		child.Stdout = os.Stdout
		child.Stderr = os.Stderr
		if err := child.Start(); err != nil {
			fail("start pipe holder: %v", err)
		}
		if err := os.WriteFile(args[0], []byte(strconv.Itoa(child.Process.Pid)), 0o600); err != nil {
			_ = child.Process.Kill()
			fail("write child pid: %v", err)
		}
		_ = child.Wait()
	case "pipe-child":
		// The test kills this child after confirming command cancellation closed
		// the parent process while this child still holds stdout and stderr.
		time.Sleep(30 * time.Second)
	default:
		fail("unknown helper mode %q", mode)
	}
}

func runFFmpeg(args []string) {
	var inputPath, outputPath string
	frames := 0
	for i, arg := range args {
		if arg == "-i" && i+1 < len(args) {
			inputPath = args[i+1]
		}
		if arg == "-frames:v" && i+1 < len(args) {
			frames, _ = strconv.Atoi(args[i+1])
		}
	}
	if len(args) > 0 {
		outputPath = args[len(args)-1]
	}
	if inputPath == "" || outputPath == "" || frames < 1 {
		fail("unexpected ffmpeg arguments: %q", args)
	}
	ivf, err := os.ReadFile(inputPath)
	if err != nil {
		fail("read IVF input: %v", err)
	}
	if len(ivf) < 16 || string(ivf[:4]) != "DKIF" {
		fail("invalid IVF header in %q", inputPath)
	}
	width := int(binary.LittleEndian.Uint16(ivf[12:14]))
	height := int(binary.LittleEndian.Uint16(ivf[14:16]))
	if width < 1 || height < 1 || width%2 != 0 || height%2 != 0 {
		fail("invalid IVF dimensions %dx%d", width, height)
	}
	decodedBytes := int64(width) * int64(height) * 3 / 2 * int64(frames)
	if err := writeZeroFile(outputPath, decodedBytes); err != nil {
		fail("write decoded output: %v", err)
	}
	if err := os.WriteFile(outputPath+".args", append([]byte(strings.Join(args, "\n")), '\n'), 0o600); err != nil {
		fail("write ffmpeg args: %v", err)
	}
}

func writeZeroFile(path string, bytes int64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	zeroes := make([]byte, 32*1024)
	for bytes > 0 {
		chunk := int64(len(zeroes))
		if bytes < chunk {
			chunk = bytes
		}
		if _, err := f.Write(zeroes[:chunk]); err != nil {
			return err
		}
		bytes -= chunk
	}
	return f.Close()
}

func fail(format string, args ...any) {
	_, _ = fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(2)
}
