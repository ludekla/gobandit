package policy

import (
	"math"
)

// Softmax is an agent that chooses arms selecting randomly from
// a Boltzmann distribution where the degree of explorativity depends
// on a parameter called temperature
type Softmax struct {
	temperature float64
}

// Constructor function. Initially the agent has no info about the
// number of arms to work on.
func NewSoftmax(temp float64) Softmax {
	return Softmax{temperature: temp}
}

// Implementation of the temperature-greedy policy.
// Chooses an arm randomly (explores) or the so far most rewarding one (exploit).
func (sm Softmax) SelectArm(values []float64) int {
	nArms := len(values)
	distro := make([]float64, nArms)
	for i, val := range values {
		distro[i] = math.Exp(val / sm.temperature)
	}
	norm := sumSlice(distro)
	for i, val := range distro {
		distro[i] = val / norm
	}
	return randIndex(distro)
}

// AnnealingSoftmax reduces the temperature and hence increases greedyness
type AnnealingSoftmax struct {
	temperature float64
	count       float64
}

// Constructor function. Initially the agent has no info about the
// number of arms to work on.
func NewAnnealingSoftmax(temp float64) AnnealingSoftmax {
	return AnnealingSoftmax{temperature: temp}
}

func (sm AnnealingSoftmax) SelectArm(values []float64) int {
	nArms := len(values)
	temp := sm.temperature / math.Log(sm.count+1.000001)
	distro := make([]float64, nArms)
	for i, val := range values {
		distro[i] = math.Exp(val / temp)
	}
	norm := sumSlice(distro)
	for i, val := range distro {
		distro[i] = val / norm
	}
	sm.count += 1.0
	return randIndex(distro)
}
