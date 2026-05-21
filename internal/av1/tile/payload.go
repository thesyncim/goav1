package tile

// PayloadRange returns the byte range in a tile-group payload occupied by job.
func (j Job) PayloadRange(payloadLen int) (int, int, error) {
	end := j.Offset + j.Size
	if payloadLen < 0 || j.Offset < 0 || j.Size < 0 || end < j.Offset || end > payloadLen {
		return 0, 0, ErrInvalidPlan
	}
	return j.Offset, end, nil
}

// Payload returns the exact tile payload bytes for job. The returned slice
// aliases payload.
func (j Job) Payload(payload []byte) ([]byte, error) {
	start, end, err := j.PayloadRange(len(payload))
	if err != nil {
		return nil, err
	}
	return payload[start:end], nil
}

// ValidatePayloads checks that every job names a valid byte range inside
// payload.
func ValidatePayloads(payload []byte, jobs []Job) error {
	for i := 0; i < len(jobs); i++ {
		if _, _, err := jobs[i].PayloadRange(len(payload)); err != nil {
			return err
		}
	}
	return nil
}
