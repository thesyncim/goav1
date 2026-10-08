//go:build !unix

package main

import "os/exec"

func configureCommandProcessGroup(*exec.Cmd) {}
