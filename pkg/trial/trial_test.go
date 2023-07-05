package trial

import (
	"testing"

	"gobandit/pkg/agent"
	"gobandit/pkg/arm"
)

var bandit = []arm.BanditArm{
	arm.NewBernoulliArm(0.3),
	arm.NewBernoulliArm(0.9),
	arm.NewBernoulliArm(0.88),
}

func TestRunEpsilonGreedy(t *testing.T) {
	ag := agent.NewEpsilonGreedy(0.2)
	fqs := Run(ag, bandit, 1000, 20)
	exp := 1
	got := agent.Argmax(fqs)
	if got != exp {
		t.Errorf("Expected arm %d, got %d", exp, got)
	}
}

func TestRunSoftmax(t *testing.T) {
	ag := agent.NewSoftmax(1.0)
	fqs := Run(ag, bandit, 1000, 20)
	exp := 1
	got := agent.Argmax(fqs)
	if got != exp {
		t.Errorf("Expected arm %d, got %d", exp, got)
	}
}

func TestRunAnnealingSoftmax(t *testing.T) {
	ag := agent.NewAnnealingSoftmax(1.0)
	fqs := Run(ag, bandit, 1000, 20)
	exp := 1
	got := agent.Argmax(fqs)
	if got != exp {
		t.Errorf("Expected arm %d, got %d", exp, got)
	}
}

func TestRunUCB(t *testing.T) {
	ag := agent.NewUCB()
	fqs := Run(ag, bandit, 1000, 20)
	exp := 1
	got := agent.Argmax(fqs)
	if got != exp {
		t.Errorf("Expected arm %d, got %d", exp, got)
	}
}

func BenchmarkEpsGreedy(b *testing.B) {
	ag := agent.NewEpsilonGreedy(0.1)
	for i := 0; i < b.N; i++ {
		Run(ag, bandit, 1000, 20)
	}
}

func BenchmarkSoftmax(b *testing.B) {
	ag := agent.NewSoftmax(1.0)
	for i := 0; i < b.N; i++ {
		Run(ag, bandit, 1000, 20)
	}
}

func BenchmarkAnnealingSoftmax(b *testing.B) {
	ag := agent.NewAnnealingSoftmax(1.0)
	for i := 0; i < b.N; i++ {
		Run(ag, bandit, 1000, 20)
	}
}

func BenchmarkUCB(b *testing.B) {
	ag := agent.NewUCB()
	for i := 0; i < b.N; i++ {
		Run(ag, bandit, 1000, 20)
	}
}
