// Package agent implements the agent part of multi-armed bandit algorithms.
package agent

import (
	"bandit/pkg/arm"
	"math"
)

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

// Agent interface represents an agent in charge conducting bandit experiments.
type Agent interface {
	Run(arms []arm.BanditArm, nEpisodes, horizon int)
	SelectArm() int
	Update(arm int, reward float64)
	Reset()
}
