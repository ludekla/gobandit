package agent

import (
	"math"
	"testing"
)

func TestArgmax(t *testing.T) {
	s := []float64{0.1, -1.3, 9.2, 5.0}
	exp := 2
	got := Argmax(s)
	if exp != got {
		t.Errorf("Expected index %d, got %d", exp, got)
	}
}

func TestSumVals(t *testing.T) {
	s := []float64{0.1, -1.3, 9.2, 5.0}
	exp := 13.0
	got := SumVals(s)
	if exp != got {
		t.Errorf("Expected index %v, got %v", exp, got)
	}
}

func TestMSE(t *testing.T) {
	s := []float64{0.1, 0.3, 0.2, 0.8}
	exp := 0.21213203435596428
	got := MSE(s, 3)
	if math.Abs(exp-got) > 1e-10 {
		t.Errorf("Expected index %v, got %v", exp, got)
	}
}
