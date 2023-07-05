package main

import (
	"flag"
	"fmt"
	"os"

	"gobandit/pkg/agent"
	"gobandit/pkg/arm"
	"gobandit/pkg/trial"
)

func main() {
	flag.Parse()
	args := flag.Args()

	parms, err := trial.GetParms("app", args)
	if err != nil {
		fmt.Printf("ERROR: %v", err)
		os.Exit(0)
	}

	bandit := make([]arm.BanditArm, len(args))
	for i, parm := range parms {
		bandit[i] = arm.NewBernoulliArm(parm)
	}

	eg := agent.NewEpsilonGreedy(0.1)
	fq := trial.Run(eg, bandit, 10000, 5)
	trial.Report("Epsilon-Greedy", bandit, fq, 2)
}
