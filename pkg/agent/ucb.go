package agent

import (
	"math"
)

// UCB implements the policy interface.
type UCB struct {
	*ProtoAgent
}

// Constructor function.
func NewUCB() UCB {
	return UCB{&ProtoAgent{}}
}

// Implementation of the epsilon-greedy policy.
// Chooses an arm randomly (explores) or the so far most rewarding one (exploit).
func (u UCB) SelectArm() int {
	for idx, count := range u.Counts {
		if count < 1.0 {
			return idx
		}
	}
	values := make([]float64, len(u.Values))
	copy(values, u.Values)
	norm := math.Log(SumVals(u.Counts))
	for i, val := range values {
		values[i] = val + math.Sqrt(norm/u.Counts[i])
	}
	return argmax(values)
}
