package agent

import (
	"bandit/pkg/arm"
)

// EpsilonGreedy is an agent that chooses arms following the epsilon-greedy policy:
// - explore: try all arms with probability epsilon
// - exploit: take the so far best arm
type EpsilonGreedy struct {
	epsilon float64
	nArms   int
	Values  []float64
	Counts  []float64
}

// Constructor function. Initially the agent has no info about the
// number of arms to work on.
func NewEpsilonGreedy(eps float64) EpsilonGreedy {
	return EpsilonGreedy{epsilon: eps}
}

// Performs bandit experiment and returns a slice of relative frequencies
// for the arms to be chosen as best after each trial of rounds. Every
// episode is given by a trial of n rounds, where n is the horizon.
func (eg *EpsilonGreedy) Run(arms []arm.BanditArm, nEpisodes, horizon int) []float64 {
	// set the number of arms and prepare the value functions
	eg.nArms = len(arms)
	eg.Values = make([]float64, eg.nArms)
	eg.Counts = make([]float64, eg.nArms)
	// frequencies of choosing the arms
	freqs := make([]float64, eg.nArms)
	// start playing
	for ep := 0; ep < nEpisodes; ep++ {
		for t := 0; t < horizon; t++ {
			idx := eg.SelectArm()
			reward := arms[idx].Draw()
			eg.Update(idx, reward)
		}
		best := argmax(eg.Values)
		// count the number of times this action comes out best in the episode
		freqs[best] += 1.0
	}
	// compute relative frequencies from absolute frequencies
	for i, res := range freqs {
		freqs[i] = res / float64(nEpisodes)
	}
	return freqs
}

// Implementation of the epsilon-greedy policy.
// Chooses an arm randomly (explores) or the so far most rewarding one (exploit).
func (eg EpsilonGreedy) SelectArm() int {
	if arm.Rng.Float64() < eg.epsilon {
		return arm.Rng.Intn(eg.nArms)
	}
	return argmax(eg.Values)
}

// Updates the action values.
func (eg *EpsilonGreedy) Update(arm int, reward float64) {
	eg.Counts[arm] += 1.0
	eg.Values[arm] += (reward - eg.Values[arm]) / eg.Counts[arm]
}

// Restores the initial situation.
func (eg *EpsilonGreedy) Reset() {
	eg.nArms = 0
	eg.Values = nil
	eg.Counts = nil
}
