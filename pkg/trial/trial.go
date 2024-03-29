package trial

import (
	"fmt"

	"gobandit/pkg/agent"
	"gobandit/pkg/arm"
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
