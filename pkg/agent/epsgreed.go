package agent

import "bandit/pkg/arm"

// EpsilonGreedy implements the poicy interface.
// It chooses arms following the epsilon-greedy policy:
// - explore: try all arms with probability epsilon
// - exploit: take the so far best arm
// ProtoAgent must be embedded as a pointer as some methods
// have a pointer receiver and would otherwise not belong to
// the method set.
type EpsilonGreedy struct {
	*ProtoAgent
	epsilon float64
}

// Constructor function.
func NewEpsilonGreedy(eps float64) EpsilonGreedy {
	return EpsilonGreedy{&ProtoAgent{}, eps}
}

// Implementation of the epsilon-greedy policy.
// Chooses an arm randomly (explores) or the so far most rewarding one (exploit).
func (eg EpsilonGreedy) SelectArm() int {
	if arm.Rng.Float64() < eg.epsilon {
		return arm.Rng.Intn(eg.nArms)
	}
	return argmax(eg.Values)
}
