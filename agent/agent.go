// Package agent implements multi-armed bandit algorithms.
package agent

import (
	"math"
	"bandit/arm"
)

func argmax(vals []float64) int {
	var max = -math.MaxFloat64
	var idx int
	for i, val := range vals {
		if val > max {
			idx = i
			max = val
		}
	}
	return idx
}

// Agent represents an agent in charge conducting bandit experiments. 
type Agent interface {
	Run(arms []arm.BanditArm, nEpisodes, horizon int)
	SelectArm() int
	Update(arm int, reward float64)
	Reset()
}

// EpsilonGreedy is an agent that chooses 
type EpsilonGreedy struct {
	epsilon float64
	nArms int
	Values []float64
	Counts []float64
}

// Constructor function
func NewEpsilonGreedy(eps float64) EpsilonGreedy {
	return EpsilonGreedy{epsilon: eps}
}

// Performs bandit experiment and returns a slice of relative frequencies
// for the arms to be chosen as best after each trial of rounds. Every 
// episode is given by a trial of n rounds, where n is the horizon.   
func (eg *EpsilonGreedy) Run(arms []arm.BanditArm, nEpisodes, horizon int) []float64 {
	eg.nArms = len(arms)
	eg.Values = make([]float64, eg.nArms)
	eg.Counts = make([]float64, eg.nArms)
	freqs := make([]float64, eg.nArms)
	for ep := 0; ep < nEpisodes; ep++ {
		for t := 0; t < horizon; t++ {
			idx := eg.SelectArm()
			reward := arms[idx].Draw()
			eg.Update(idx, reward)
		}
		best := argmax(eg.Values)
		freqs[best] += 1.0
	}
	for i, res := range freqs {
		freqs[i] = res / float64(nEpisodes)
	}
	return freqs
}

// Chooses an arm randomly (explores) or the so far most rewarding one
func (eg EpsilonGreedy) SelectArm() int {
	if arm.Rng.Float64() < eg.epsilon {
		return arm.Rng.Intn(eg.nArms)
	}
	return argmax(eg.Values)
}

// Updates the action values
func (eg *EpsilonGreedy) Update(arm int, reward float64) {
	eg.Counts[arm] += 1.0
	eg.Values[arm] += (reward - eg.Values[arm]) / eg.Counts[arm]
} 

// Restores the initial situation
func (eg *EpsilonGreedy) Reset() {
	eg.nArms = 0
	eg.Values = nil
	eg.Counts = nil
}
