// Implements the bandit's arms.
package arm

import (
	"math/rand"
)

// BanditArm returns a reward when drawn.
type BanditArm interface {
	Draw() float64
}

// BernoulliArm implements the BanditArm interface and acts as a simple
// Bernoulli random variable. It yields a value of 1.0 as reward with
// probablilty p, and 0.0 otherwise.
type BernoulliArm struct {
	p float64
}

// Constructor for BernoulliArm.
func NewBernoulliArm(prob float64) BernoulliArm {
	return BernoulliArm{p: prob}
}

// Implements a Bernoulli random variable with parameter p.
func (b BernoulliArm) Draw() float64 {
	if rand.Float64() < b.p {
		return 1.0
	}
	return 0.0
}
