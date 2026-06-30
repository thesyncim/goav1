package encoder

// The realtime SVC controller in libaom carries cumulative temporal-layer
// targets and framerate decimators in the layer context:
//
//	av1/encoder/svc_layercontext.c:av1_update_temporal_layer_framerate
//	test/ratectrl_rtc_test.cc:SetConfigSvc
var realtimeSVCTemporalRateAllocation = [WebRTCMaxTemporalLayers][WebRTCMaxTemporalLayers]int{
	{100, 0, 0},
	{60, 100, 0},
	{50, 70, 100},
}

var realtimeSVCTemporalFramerateFactor = [WebRTCMaxTemporalLayers][WebRTCMaxTemporalLayers]int{
	{1, 0, 0},
	{2, 1, 0},
	{4, 2, 1},
}

func rateControlTemporalLayerIndex(temporalLayers int, temporalID uint8) int {
	if temporalLayers < 1 {
		temporalLayers = 1
	}
	if temporalLayers > WebRTCMaxTemporalLayers {
		temporalLayers = WebRTCMaxTemporalLayers
	}
	idx := int(temporalID)
	if idx < 0 || idx >= temporalLayers {
		return 0
	}
	return idx
}

func resetRateControlTemporalState(qIndex uint8, targetBitsPerSecond int, framesPerSecond int, temporalLayers int, fallbackPerFrameBits int, q *[WebRTCMaxTemporalLayers]uint8, buffers *[WebRTCMaxTemporalLayers]int, recent *[WebRTCMaxTemporalLayers][2]int, perFrameBits *[WebRTCMaxTemporalLayers]int) {
	for i := range q {
		q[i] = qIndex
		buffers[i] = 0
		recent[i] = [2]int{}
		perFrameBits[i] = fallbackPerFrameBits
	}
	if temporalLayers < 1 {
		temporalLayers = 1
	}
	if temporalLayers > WebRTCMaxTemporalLayers {
		temporalLayers = WebRTCMaxTemporalLayers
	}
	if targetBitsPerSecond <= 0 || framesPerSecond <= 0 {
		return
	}
	for tl := 0; tl < temporalLayers; tl++ {
		perFrameBits[tl] = rateControlTemporalLayerPerFrameBits(targetBitsPerSecond, framesPerSecond, temporalLayers, uint8(tl), fallbackPerFrameBits)
	}
}

func applyRateControlConfigPreservingState(rc RateControlConfig, temporalLayers int, surplusFrameLimit int, qIndex *uint8, targetBitsPerSecond *int, framesPerSecond *int, perFrameBits *int, minQ *uint8, maxQ *uint8, buffer *int, temporalQ *[WebRTCMaxTemporalLayers]uint8, temporalBuffers *[WebRTCMaxTemporalLayers]int, temporalPerFrameBits *[WebRTCMaxTemporalLayers]int) error {
	nextPerFrameBits, err := rateControlPerFrameBits(rc)
	if err != nil {
		return err
	}
	*targetBitsPerSecond = rc.TargetBitsPerSecond
	*framesPerSecond = rc.FramesPerSecond
	*perFrameBits = nextPerFrameBits
	*minQ = rc.MinQIndex
	*maxQ = rc.MaxQIndex
	*qIndex = clampRateControlQIndex(*qIndex, *minQ, *maxQ)
	*buffer = clampRateControlBuffer(*buffer, nextPerFrameBits, surplusFrameLimit)
	updateRateControlTemporalBudgetsPreservingState(*qIndex, rc.TargetBitsPerSecond, rc.FramesPerSecond, temporalLayers, nextPerFrameBits, surplusFrameLimit, *minQ, *maxQ, temporalQ, temporalBuffers, temporalPerFrameBits)
	return nil
}

func updateRateControlTemporalBudgetsPreservingState(qIndex uint8, targetBitsPerSecond int, framesPerSecond int, temporalLayers int, fallbackPerFrameBits int, surplusFrameLimit int, minQ uint8, maxQ uint8, q *[WebRTCMaxTemporalLayers]uint8, buffers *[WebRTCMaxTemporalLayers]int, perFrameBits *[WebRTCMaxTemporalLayers]int) {
	for i := range perFrameBits {
		perFrameBits[i] = fallbackPerFrameBits
	}
	if temporalLayers < 1 {
		temporalLayers = 1
	}
	if temporalLayers > WebRTCMaxTemporalLayers {
		temporalLayers = WebRTCMaxTemporalLayers
	}
	for tl := 0; tl < temporalLayers; tl++ {
		if q[tl] == 0 {
			q[tl] = qIndex
		}
		q[tl] = clampRateControlQIndex(q[tl], minQ, maxQ)
		perFrameBits[tl] = rateControlTemporalLayerPerFrameBits(targetBitsPerSecond, framesPerSecond, temporalLayers, uint8(tl), fallbackPerFrameBits)
	}
	for i := range buffers {
		buffers[i] = clampRateControlBuffer(buffers[i], perFrameBits[i], surplusFrameLimit)
	}
}

func rateControlTemporalLayerPerFrameBits(targetBitsPerSecond int, framesPerSecond int, temporalLayers int, temporalID uint8, fallback int) int {
	if targetBitsPerSecond <= 0 || framesPerSecond <= 0 {
		if fallback > 0 {
			return fallback
		}
		return 1
	}
	if temporalLayers <= 1 {
		if fallback > 0 {
			return fallback
		}
		bits := targetBitsPerSecond / framesPerSecond
		if bits < 1 {
			bits = 1
		}
		return bits
	}
	if temporalLayers > WebRTCMaxTemporalLayers {
		temporalLayers = WebRTCMaxTemporalLayers
	}
	tl := rateControlTemporalLayerIndex(temporalLayers, temporalID)
	allocation := realtimeSVCTemporalRateAllocation[temporalLayers-1]
	factors := realtimeSVCTemporalFramerateFactor[temporalLayers-1]
	layerTarget := targetBitsPerSecond * allocation[tl] / 100
	factor := factors[tl]
	if tl == 0 {
		bits := divRoundNearest(layerTarget*factor, framesPerSecond)
		if bits < 1 {
			bits = 1
		}
		return bits
	}
	prevTarget := targetBitsPerSecond * allocation[tl-1] / 100
	prevFactor := factors[tl-1]
	bits := divRoundNearest((layerTarget-prevTarget)*factor*prevFactor, framesPerSecond*(prevFactor-factor))
	if bits < 1 {
		bits = 1
	}
	return bits
}

func divRoundNearest(num int, den int) int {
	if den <= 0 {
		return 0
	}
	if num >= 0 {
		return (num + den/2) / den
	}
	return -((-num + den/2) / den)
}

func clampRateControlQIndex(q uint8, minQ uint8, maxQ uint8) uint8 {
	if q < minQ {
		return minQ
	}
	if q > maxQ {
		return maxQ
	}
	return q
}

func clampRateControlBuffer(buffer int, perFrameBits int, surplusFrameLimit int) int {
	if perFrameBits <= 0 {
		perFrameBits = 1
	}
	if surplusFrameLimit < 1 {
		surplusFrameLimit = 1
	}
	minBuffer := -24 * perFrameBits
	if buffer < minBuffer {
		return minBuffer
	}
	maxBuffer := surplusFrameLimit * perFrameBits
	if buffer > maxBuffer {
		return maxBuffer
	}
	return buffer
}

func rcUpdateState(frameBits int, perFrameBits int, surplusFrameLimit int, minQ uint8, maxQ uint8, qIndex uint8, buffer *int, recent *[2]int) uint8 {
	if perFrameBits <= 0 {
		perFrameBits = 1
	}
	recent[0], recent[1] = recent[1], frameBits
	*buffer += perFrameBits - frameBits
	if limit := 24 * perFrameBits; *buffer < -limit {
		*buffer = -limit
	}
	if surplusFrameLimit < 1 {
		surplusFrameLimit = 1
	}
	if limit := surplusFrameLimit * perFrameBits; *buffer > limit {
		*buffer = limit
	}
	q := int(qIndex)
	step := -*buffer * 4 / perFrameBits
	if step > 12 {
		step = 12
	} else if step < -12 {
		step = -12
	}
	q += step
	if q < int(minQ) {
		q = int(minQ)
	}
	if q > int(maxQ) {
		q = int(maxQ)
	}
	return uint8(q)
}

func rcKeyframeQIndex(qIndex uint8, rcEnabled bool, perFrameBits int, buffer int, minQ uint8, maxQ uint8) uint8 {
	keyQ := qIndex
	if !rcEnabled {
		return keyQ
	}
	if perFrameBits <= 0 {
		perFrameBits = 1
	}
	boost := perFrameBits / 1600
	if boost > 50 {
		boost = 50
	} else if boost < 12 {
		boost = 12
	}
	if buffer < 0 {
		horizon := 24 * perFrameBits
		credit := horizon + buffer
		if credit < 0 {
			credit = 0
		}
		boost = boost * (3*horizon + credit) / (4 * horizon)
	}
	if int(keyQ)-boost > int(minQ) {
		keyQ -= uint8(boost)
	} else {
		keyQ = minQ
	}
	if buffer >= 0 {
		maxKeyQ := int(minQ) + (int(maxQ)-int(minQ))*2/3
		if int(keyQ) > maxKeyQ {
			keyQ = uint8(maxKeyQ)
		}
	}
	return keyQ
}

func rateControlTemporalLayerQIndex(qs [WebRTCMaxTemporalLayers]uint8, minQ uint8, maxQ uint8, temporalLayers int, temporalID uint8) uint8 {
	idx := rateControlTemporalLayerIndex(temporalLayers, temporalID)
	q := int(qs[idx])
	if idx > 0 {
		// Mirrors the CBR SVC clamp in ratectrl.c:adjust_q_cbr that prevents an
		// enhancement layer from going more than four qindex units below TL0.
		tl0Floor := int(qs[0]) - 4
		if q < tl0Floor {
			q = tl0Floor
		}
	}
	if q < int(minQ) {
		q = int(minQ)
	}
	if q > int(maxQ) {
		q = int(maxQ)
	}
	return uint8(q)
}
