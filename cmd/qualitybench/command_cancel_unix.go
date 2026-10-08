//go:build unix

package main

import (
	"errors"
	"os"
	"os/exec"
	"syscall"
)

func configureCommandProcessGroup(cmd *exec.Cmd) {
	// External tools can start helpers that inherit stdout/stderr. Put the command
	// in its own process group so a timeout can stop the whole process tree.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return os.ErrProcessDone
		}
		err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		if errors.Is(err, syscall.ESRCH) {
			return os.ErrProcessDone
		}
		return err
	}
}
