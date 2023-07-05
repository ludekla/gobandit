package agent

import (
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
