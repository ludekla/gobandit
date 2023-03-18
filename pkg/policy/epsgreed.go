package policy

import (
	"bandit/pkg/arm"
)

// EpsilonGreedy implements the poicy interface.
// It chooses arms following the epsilon-greedy policy:
// - explore: try all arms with probability epsilon
// - exploit: take the so far best arm
type EpsilonGreedy struct {
	epsilon float64
}

// Constructor function.
func NewEpsilonGreedy(eps float64) EpsilonGreedy {
	return EpsilonGreedy{epsilon: eps}
}

// Implementation of the epsilon-greedy policy.
// Chooses an arm randomly (explores) or the so far most rewarding one (exploit).
func (eg EpsilonGreedy) SelectArm(values []float64) int {
	nArms := len(values)
	if arm.Rng.Float64() < eg.epsilon {
		return arm.Rng.Intn(nArms)
	}
	return Argmax(values)
}
