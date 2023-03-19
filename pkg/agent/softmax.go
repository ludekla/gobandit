package agent

import (
	"math"
)

// Softmax is an agent that chooses arms selecting randomly from
// a Boltzmann distribution where the degree of explorativity depends
// on a parameter called temperature.
type Softmax struct {
	*ProtoAgent
	temperature float64
}

// Constructor function. Initially the agent has no info about the
// number of arms to work on.
func NewSoftmax(temp float64) Softmax {
	return Softmax{&ProtoAgent{}, temp}
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

// AnnealingSoftmax reduces the temperature and hence increases greedyness
type AnnealingSoftmax struct {
	*ProtoAgent
	temperature float64
}

// Constructor function. Initially the agent has no info about the
// number of arms to work on.
func NewAnnealingSoftmax(temp float64) AnnealingSoftmax {
	return AnnealingSoftmax{&ProtoAgent{}, temp}
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
