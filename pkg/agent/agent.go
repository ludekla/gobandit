// Package agent implements the agent part of multi-armed bandit algorithms.
package agent

import (
	"bandit/pkg/arm"
	"math"
)

// Agent interface represents an agent in charge conducting bandit experiments.
type Agent interface {
	Run(arms []arm.BanditArm, nEpisodes, horizon int)
	SelectArm() int
	Update(arm int, reward float64)
	Reset()
}

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
			sum += (val - 1.0)*(val - 1.0)
		} else {
			sum += val*val
		}
	}
	return math.Sqrt(sum / float64(len(distro)))
}
