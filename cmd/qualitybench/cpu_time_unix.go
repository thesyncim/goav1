//go:build unix

package main

import (
	"syscall"
	"time"
)

func currentProcessCPUTimes() (processCPUTimes, bool) {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return processCPUTimes{}, false
	}
	return processCPUTimes{
		user:   time.Duration(syscall.TimevalToNsec(usage.Utime)),
		system: time.Duration(syscall.TimevalToNsec(usage.Stime)),
	}, true
}
