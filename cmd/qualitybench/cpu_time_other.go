//go:build !unix

package main

func currentProcessCPUTimes() (processCPUTimes, bool) {
	return processCPUTimes{}, false
}
