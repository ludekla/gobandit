package agent

import "math/rand"

// EpsilonGreedy implements the Agent interface.
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

// Constructor function for the EpsilonGreedy agent.
func NewEpsilonGreedy(eps float64) EpsilonGreedy {
	return EpsilonGreedy{&ProtoAgent{}, eps}
}

// Implementation of the epsilon-greedy policy.
// SelectArm chooses an arm randomly (explores) or the so far most
// rewarding one (exploit - greedy choice).
func (eg EpsilonGreedy) SelectArm() int {
	if rand.Float64() < eg.epsilon {
		return rand.Intn(eg.nArms)
	}
	return argmax(eg.Values)
}
