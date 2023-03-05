package main

import (
	"bandit/arm"
	"bandit/agent"
	"fmt"
)

func main() {
	bandit := []arm.BanditArm{
		arm.NewBernoulliArm(0.2),
		arm.NewBernoulliArm(0.5),
		arm.NewBernoulliArm(0.3),
		arm.NewBernoulliArm(0.4),
	}

	eg := agent.NewEpsilonGreedy(0.3)
	fq := eg.Run(bandit, 1000, 5)

	fmt.Printf("Bandit: %v, Freqs: %v\n", bandit, fq)
}