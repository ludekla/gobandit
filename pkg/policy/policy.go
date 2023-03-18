package policy

import (
	"math/rand"
	"math"
)

type Policy interface {
	SelectArm(values []float64) int
}

// Helper functions
// Finds the maximum value of the slice and returns its slice index
func Argmax(vals []float64) int {
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
