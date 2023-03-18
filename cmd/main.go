package main

import (
    "fmt"
    "bandit/pkg/arm"
    "bandit/pkg/agent"
	"bandit/pkg/policy"
)

func Report(name string, bandit []arm.BanditArm, fq []float64, idx int) {
	fmt.Println(name)
	fmt.Printf("Bandit: %v\n", bandit)
	fmt.Printf("Freqs:  %v\n", fq)
	fmt.Printf("Error:  %v\n", agent.MSE(fq, idx))
}

func main() {
    bandit := []arm.BanditArm{
		arm.NewBernoulliArm(0.1),
		arm.NewBernoulliArm(0.1),
		arm.NewBernoulliArm(0.15),
		arm.NewBernoulliArm(0.1),
	}

	eg := policy.NewEpsilonGreedy(0.1)
	egPlayer := agent.NewPlayer(eg)
	fq := egPlayer.Run(bandit, 10000, 5)
	Report("Epsilon-Greedy", bandit, fq, 2)

	sm := policy.NewSoftmax(1.0)
	smPlayer := agent.NewPlayer(sm)
	fq = smPlayer.Run(bandit, 10000, 5)
	Report("Softmax", bandit, fq, 2)

    asm := policy.NewAnnealingSoftmax(1.0)
	asmPlayer := agent.NewPlayer(asm)
	fq = asmPlayer.Run(bandit, 10000, 5)
	Report("Annealing Softmax", bandit, fq, 2)
}