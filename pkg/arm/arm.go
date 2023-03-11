// Implements the bandit's arms.
package arm

import (
	"math/rand"
	"time"
)

// Random number generator for whatever package 
// or source code within this project needs it.  
var Rng *rand.Rand

// Set up random number generator and seed it with unix time.
func init() {
	var seed = time.Now().UnixNano()
	Rng = rand.New(rand.NewSource(seed))
}

// BanditArm returns a reward when drawn or pulled.
type BanditArm interface {
	Draw() float64
}

// Arm acts as a simple Bernoulli random variable 
// yielding a value of 1.0 as reward with probablilty p, 
// and 0.0 otherwise.
type BernoulliArm struct {
	p float64
}

// Constructor.
func NewBernoulliArm(prob float64) BernoulliArm {
	return BernoulliArm{p: prob}
}

// Implements a Bernoulli random variable with parameter p.
func (b BernoulliArm) Draw() float64 {
	if Rng.Float64() < b.p {
		return 1.0
	}
	return 0.0
}