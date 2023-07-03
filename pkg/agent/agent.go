// Implements the agent part of multi-armed bandit algorithms.
package agent

import (
	"math"
	"math/rand"
)

// Helper functions
// argmax finds the maximum value of the slice and returns its slice index.
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

// SumVals computes the sum of the values of a slice.
func SumVals(vals []float64) float64 {
	var sum float64
	for _, val := range vals {
		sum += val
	}
	return sum
}

// randIndex returns a random index (int) by probing the given distribution.
// The values of distro must sum up to unity. The returned index represents the
// the arm the arm that has been chosen.
func randIndex(distro []float64) int {
	var cumsum float64 // cumulative sum
	val := rand.Float64()
	for i, prob := range distro {
		cumsum += prob
		if cumsum > val {
			return i
		}
	}
	return len(distro) - 1
}

// MSE computes the mean-squared error between the given distribution distro
// and the 'correct' distribution with its weight sitting on one index (idx).
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

// ProtoAgent implements part of the Agent interface with the purpose of being
// embedded into fully-fledged agents that implement the SelectArm method.
type ProtoAgent struct {
	nArms  int
	Values []float64
	Counts []float64
}

// Init initialises the agent by setting up all-zero slices.
func (pa *ProtoAgent) Init(nArms int) {
	pa.nArms = nArms
	pa.Values = make([]float64, nArms)
	pa.Counts = make([]float64, nArms)
}

// Update updates the action values.
func (pa *ProtoAgent) Update(arm int, reward float64) {
	pa.Counts[arm] += 1.0
	pa.Values[arm] += (reward - pa.Values[arm]) / pa.Counts[arm]
}

// BestAction returns the action with the so far best value (greedy choice).
func (pa ProtoAgent) BestAction() int {
	return argmax(pa.Values)
}
