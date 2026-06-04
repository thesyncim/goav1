package tile

// PayloadRange returns the byte range in a tile-group payload occupied by job.
func (j Job) PayloadRange(payloadLen int) (int, int, error) {
	end := uint64(j.Offset) + uint64(j.Size)
	maxInt := uint64(^uint(0) >> 1)
	if payloadLen < 0 || end > uint64(payloadLen) || end > maxInt {
		return 0, 0, ErrInvalidPlan
	}
	return int(j.Offset), int(end), nil
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
	for i := range jobs {
		if _, _, err := jobs[i].PayloadRange(len(payload)); err != nil {
			return err
		}
	}
	return nil
}
