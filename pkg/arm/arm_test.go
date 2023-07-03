package arm

import (
	"math/rand"
	"testing"
)

// Simple Test for BernoulliArm for fixed seed.
func TestBernoulliArm(t *testing.T) {
	rand.Seed(0)
	p := 0.5
	barm := NewBernoulliArm(p)
	if barm.p != p {
		t.Errorf("Expected %v as Bernoulli parameter, got %v", p, barm.p)
	}
	for _, expected := range []float64{0, 1, 0, 1, 1, 1, 1, 0, 0, 1} {
		got := barm.Draw()
		if got != expected {
			t.Errorf("Expected %v, got %v", expected, got)
		}
	}
}
