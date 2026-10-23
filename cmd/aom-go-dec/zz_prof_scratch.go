//go:build goav1_scratchprof

package main

import (
	"fmt"
	"os"
	"runtime/pprof"
	"strconv"
)

func init() {
	p := os.Getenv("GOAV1_CPUPROF")
	if p == "" {
		return
	}
	n, _ := strconv.Atoi(os.Getenv("GOAV1_REPS"))
	if n < 1 {
		n = 1
	}
	f, _ := os.Create(p)
	pprof.StartCPUProfile(f)
	for i := 0; i < n; i++ {
		if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
			fmt.Fprintln(os.Stderr, err)
		}
	}
	pprof.StopCPUProfile()
	f.Close()
	os.Exit(0)
}
