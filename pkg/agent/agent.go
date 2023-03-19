// Package agent implements the agent part of multi-armed bandit algorithms.
package agent

import (
	"math"
	"math/rand"
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

// Agent interface represents an agent conducting bandit experiments.
type Agent interface {
	Init(nArms int)
	SelectArm() int
	Update(arm int, reward float64)
	BestAction() int
}

// ProtoAgent is implements part of the Agent interface with the purpose of being 
// embedded into fully-fledged agents that implement the selectArm method.
type ProtoAgent struct {
	nArms  int
	Values []float64
	Counts []float64
}

// Initialises the agent.
func (pa *ProtoAgent) Init(nArms int) {
	pa.nArms = nArms
	pa.Values = make([]float64, nArms)
	pa.Counts = make([]float64, nArms)
}

// Updates the action values.
func (pa *ProtoAgent) Update(arm int, reward float64) {
	pa.Counts[arm] += 1.0
	pa.Values[arm] += (reward - pa.Values[arm]) / pa.Counts[arm]
}

// Returns the action with best value.
func (pa ProtoAgent) BestAction() int {
	return argmax(pa.Values)
}


