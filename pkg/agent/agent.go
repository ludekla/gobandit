// Package agent implements the agent part of multi-armed bandit algorithms.
package agent

import (
	"bandit/pkg/arm"
	"math"
)

// Helper functions
// Finds the maximum value of the slice and returns its slice index
func argmax(vals []float64) int {
	var max = -math.MaxFloat64 // smallest possible float64
	var idx int                // index to be returned
	// iterates through slice to find maximum value
	for i, val := range vals {
		if val > max {
			idx = i
			max = val
		}
	}
	return idx
}

// Computes mean-squared error of a distro to the delta-distro
// characterised by die index that carries the weight
func MSE(distro []float64, idx int) float64 {
	var sum float64
	for i, val := range distro {
		if i == idx {
			sum += (val - 1.0) * (val - 1.0)
		} else {
			sum += val * val
		}
	}
	return math.Sqrt(sum / float64(len(distro)))
}

// Agent interface represents an agent in charge conducting bandit experiments.
type Agent interface {
	Run(arms []arm.BanditArm, nEpisodes, horizon int)
	SelectArm() int
	Update(arm int, reward float64)
	Reset()
}

// Implementation of Agen Interface
type Player struct {
	nArms  int
	Values []float64
	Counts []float64
}

// Performs bandit experiment and returns a slice of relative frequencies
// for the arms to be chosen as best after each trial of rounds. Every
// episode is given by a trial of n rounds, where n is the horizon.
func (pl *Player) Run(arms []arm.BanditArm, nEpisodes, horizon int) []float64 {
	// set the number of arms and prepare the value functions
	pl.nArms = len(arms)
	pl.Values = make([]float64, pl.nArms)
	pl.Counts = make([]float64, pl.nArms)
	// frequencies of choosing the arms
	freqs := make([]float64, pl.nArms)
	// start playing
	for ep := 0; ep < nEpisodes; ep++ {
		for t := 0; t < horizon; t++ {
			idx := pl.SelectArm()
			reward := arms[idx].Draw()
			pl.Update(idx, reward)
		}
		best := argmax(pl.Values)
		// count the number of times this action comes out best in the episode
		freqs[best] += 1.0
	}
	// compute relative frequencies from absolute frequencies
	for i, res := range freqs {
		freqs[i] = res / float64(nEpisodes)
	}
	return freqs
}

// Implementation of the temperature-greedy policy.
// Chooses an arm randomly (explores) or the so far most rewarding one (exploit).
func (pl Player) SelectArm() int {
	return 0
}

// Updates the action values.
func (pl *Player) Update(arm int, reward float64) {
	pl.Counts[arm] += 1.0
	pl.Values[arm] += (reward - pl.Values[arm]) / pl.Counts[arm]
}

// Restores the initial situation.
func (pl *Player) Reset() {
	pl.nArms = 0
	pl.Values = nil
	pl.Counts = nil
}
