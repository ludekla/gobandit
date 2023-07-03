package main

import (
    "fmt"
    "flag"
    "os"
    "strconv"

    "gobandit/pkg/arm"
    "gobandit/pkg/agent"
)

// Performs the bandit experiment and returns a slice of relative frequencies
// for the arms that have been chosen as best after each trial of rounds. Every
// episode is given by a trial of n rounds, where n is the horizon.
func Run(ag agent.Agent, arms []arm.BanditArm, nEpisodes, horizon int) []float64 {
	// set the number of arms and prepare the value functions
	nArms := len(arms)
	ag.Init(nArms)
	// frequencies of choosing the arms
	freqs := make([]float64, nArms)
	// start playing
	for ep := 0; ep < nEpisodes; ep++ {
		for t := 0; t < horizon; t++ {
			idx := ag.SelectArm()
			reward := arms[idx].Draw()
			ag.Update(idx, reward)
		}
		best := ag.BestAction()
		// count the number of times this action comes out best in the episode
		freqs[best] += 1.0
	}
	// compute relative frequencies from absolute frequencies
	for i, res := range freqs {
		freqs[i] = res / float64(nEpisodes)
	}
	return freqs
}

func Report(name string, bandit []arm.BanditArm, fq []float64, idx int) {
	fmt.Println(name)
	fmt.Printf("Bandit: %v\n", bandit)
	fmt.Printf("Freqs:  %v\n", fq)
	fmt.Printf("Error:  %v\n", agent.MSE(fq, idx))
}

func main() {

    flag.Parse()
    args := flag.Args()
    app := "app"

    if len(args) < 2 {
        fmt.Println("Please provide at least 2 numbers between 0 and 1 as bandit-arm parameters.")
        fmt.Printf("Usage bin/%s [...numbers]\n", app)
        os.Exit(0)
    }

    parms := make([]float64, len(args))
    for i, strval := range args {
        val, err := strconv.ParseFloat(strval, 64)
        if err != nil {
            fmt.Printf("Cannot interpret %q as float. Try again.\n", strval)
            os.Exit(0)
        } else if val < 0.0 || val > 1.0 {
            fmt.Printf("Cannot interpret %q as float in [0, 1]. Try again.\n", strval)
            os.Exit(0)
        }
        parms[i] = val
    }

    bandit := make([]arm.BanditArm, len(args))
    for i, parm := range parms {
		bandit[i] = arm.NewBernoulliArm(parm)
	}

	eg := agent.NewEpsilonGreedy(0.1)
	fq := Run(eg, bandit, 10000, 5)
	Report("Epsilon-Greedy", bandit, fq, 2)

	sm := agent.NewSoftmax(1.0)
	fq = Run(sm, bandit, 10000, 5)
	Report("Softmax", bandit, fq, 2)

    am := agent.NewAnnealingSoftmax(1.0)
	fq = Run(am, bandit, 10000, 5)
	Report("Annealing Softmax", bandit, fq, 2)

	uc := agent.NewUCB()
	fq = Run(uc, bandit, 10000, 5)
	Report("UCB", bandit, fq, 2)
}
