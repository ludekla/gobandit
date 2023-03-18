package agent

import (
	"bandit/pkg/arm"
	"math"
	"math/rand"
)

// Helper functions
// Computes the sum of the values of a slice
func sumSlice(vals []float64) float64 {
	var sum float64
	for _, val := range vals {
		sum += val
	}
	return sum
}

// Returns a random int by probing the given distribution
func randIndex(distro []float64) int {
	var cumsum float64
	val := rand.Float64()
	for i, prob := range distro {
		cumsum += prob
		if cumsum > val {
			return i
		}
	}
	return len(distro) - 1
}

// Softmax is an agent that chooses arms selecting randomly from
// a Boltzmann distribution where the degree of explorativity depends
// on a parameter called temperature
type Softmax struct {
	temperature float64
	nArms       int
	Values      []float64
	Counts      []float64
}

// Constructor function. Initially the agent has no info about the
// number of arms to work on.
func NewSoftmax(temp float64) Softmax {
	return Softmax{temperature: temp}
}

// Performs bandit experiment and returns a slice of relative frequencies
// for the arms to be chosen as best after each trial of rounds. Every
// episode is given by a trial of n rounds, where n is the horizon.
func (sm *Softmax) Run(arms []arm.BanditArm, nEpisodes, horizon int) []float64 {
	// set the number of arms and prepare the value functions
	sm.nArms = len(arms)
	sm.Values = make([]float64, sm.nArms)
	sm.Counts = make([]float64, sm.nArms)
	// frequencies of choosing the arms
	freqs := make([]float64, sm.nArms)
	// start playing
	for ep := 0; ep < nEpisodes; ep++ {
		for t := 0; t < horizon; t++ {
			idx := sm.SelectArm()
			reward := arms[idx].Draw()
			sm.Update(idx, reward)
		}
		best := argmax(sm.Values)
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
func (sm Softmax) SelectArm() int {
	distro := make([]float64, sm.nArms)
	for i, val := range sm.Values {
		distro[i] = math.Exp(val / sm.temperature)
	}
	norm := sumSlice(distro)
	for i, val := range distro {
		distro[i] = val / norm
	}
	return randIndex(distro)
}

// Updates the action values.
func (sm *Softmax) Update(arm int, reward float64) {
	sm.Counts[arm] += 1.0
	sm.Values[arm] += (reward - sm.Values[arm]) / sm.Counts[arm]
}

// Restores the initial situation.
func (sm *Softmax) Reset() {
	sm.nArms = 0
	sm.Values = nil
	sm.Counts = nil
}

// AnnealingSoftmax reduces the temperature and hence increases greedyness
type AnnealingSoftmax struct {
	Softmax
}

// Constructor function. Initially the agent has no info about the
// number of arms to work on.
func NewAnnealingSoftmax(temp float64) AnnealingSoftmax {
	return AnnealingSoftmax{Softmax{temperature: temp}}
}

func (sm AnnealingSoftmax) SelectArm() int {
	temp := sm.temperature / math.Log(sumSlice(sm.Counts)+1.000001)
	distro := make([]float64, sm.nArms)
	for i, val := range sm.Values {
		distro[i] = math.Exp(val / temp)
	}
	norm := sumSlice(distro)
	for i, val := range distro {
		distro[i] = val / norm
	}
	return randIndex(distro)
}
