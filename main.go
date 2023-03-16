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

	fmt.Println("Softmax")
	fmt.Printf("Bandit: %v\n", bandit)
	fmt.Printf("Freqs:  %v\n", fq1)
	fmt.Printf("Error:  %v\n", agent.MSE(fq1, 2))

    asm := agent.NewAnnealingSoftmax(1.0)
	fq2 := asm.Run(bandit, 10000, 5)

	fmt.Println("Annealing Softmax")
	fmt.Printf("Bandit: %v\n", bandit)
	fmt.Printf("Freqs:  %v\n", fq2)
	fmt.Printf("Error:  %v\n", agent.MSE(fq2, 2))
}