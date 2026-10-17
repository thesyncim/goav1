package encoder

import "testing"

func TestPFrameDecideWorkerAdoptKeepsLaneSubpelCache(t *testing.T) {
	ownerCache := &subpelProbeCache{}
	src := &lossyEncodeState{subpelCache: ownerCache}
	var workers [3]pframeDecideWorker
	for i := range workers {
		workers[i].st.subpelCache = &subpelProbeCache{}
		workers[i].adopt(src)
	}

	for i := range workers {
		cache := workers[i].st.subpelCache
		if cache == nil || cache == ownerCache {
			t.Fatalf("lane %d did not keep a private cache", i)
		}
		for j := 0; j < i; j++ {
			if cache == workers[j].st.subpelCache {
				t.Fatalf("lanes %d and %d share a subpel cache", j, i)
			}
		}
	}

	workers[0].st.subpelCache.imValid[0] = true
	workers[0].adopt(src)
	if !workers[0].st.subpelCache.imValid[0] {
		t.Fatal("repeated adopt discarded lane-owned subpel cache state")
	}
	if ownerCache.imValid[0] {
		t.Fatal("lane cache mutation changed the owner cache")
	}
}
