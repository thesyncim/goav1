package goav1

import "syscall"

// threadCPU returns the calling OS thread's CPU time in arbitrary but fixed
// units (Mach absolute-time ticks), via the thread_selfusage system call that
// backs clock_gettime(CLOCK_THREAD_CPUTIME_ID). Callers compare ratios, so
// the unit does not matter. The caller must hold runtime.LockOSThread.
func threadCPU() (uint64, bool) {
	const sysThreadSelfusage = 482
	r, _, e := syscall.Syscall(sysThreadSelfusage, 0, 0, 0)
	if e != 0 {
		return 0, false
	}
	return uint64(r), true
}
