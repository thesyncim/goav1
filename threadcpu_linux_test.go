package goav1

import "syscall"

// threadCPU returns the calling OS thread's user+system CPU time in
// microseconds (getrusage RUSAGE_THREAD). The caller must hold
// runtime.LockOSThread.
func threadCPU() (uint64, bool) {
	const rusageThread = 1
	var ru syscall.Rusage
	if err := syscall.Getrusage(rusageThread, &ru); err != nil {
		return 0, false
	}
	return uint64(ru.Utime.Sec+ru.Stime.Sec)*1e6 + uint64(ru.Utime.Usec+ru.Stime.Usec), true
}
