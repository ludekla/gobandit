// Implements the bandit's arms.
package arm

import (
	"math/rand"
	"time"
)

var Rng *rand.Rand

func init() {
	var seed = time.Now().UnixNano()
	Rng = rand.New(rand.NewSource(seed))
}

// BanditArm returns a reward when drawn or pulled.
type BanditArm interface {
	Draw() float64
}

// BernoulliArm acts as a simple Bernoulli random variable 
// yielding a value of 1.0 as reward with probablilty p, 
// and 0.0 otherwise.
type BernoulliArm struct {
	p float64
}

func NewBernoulliArm(prob float64) BernoulliArm {
	return BernoulliArm{p: prob}
}

func (b BernoulliArm) Draw() float64 {
	if Rng.Float64() < b.p {
		return 1.0
	}
	return 0.0
}