package main

import (
    "fmt"
    "bandit/pkg/arm"
    "bandit/pkg/agent"
)

func main() {
    bandit := []arm.BanditArm{
		arm.NewBernoulliArm(0.1),
		arm.NewBernoulliArm(0.1),
		arm.NewBernoulliArm(0.15),
		arm.NewBernoulliArm(0.1),
	}

	sm := agent.NewSoftmax(1.0)
	fq1 := sm.Run(bandit, 10000, 5)

	fmt.Printf("Bandit: %v, Freqs: %v\n", bandit, fq1)

    asm := agent.NewAnnealingSoftmax(1.0)
	fq2 := asm.Run(bandit, 10000, 5)

	fmt.Printf("Bandit: %v, Freqs: %v\n", bandit, fq2)
}