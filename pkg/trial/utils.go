package trial

import (
	"fmt"
	"strconv"
)

func GetParms(app string, args []string) ([]float64, error) {
	if len(args) < 2 {
		msg := "Please provide at least 2 numbers between 0 and 1 as bandit-arm parameters.\n"
		return nil, fmt.Errorf("%s\nUsage bin/%s [...numbers]\n", msg, app)
	}
	parms := make([]float64, len(args))
	for i, strval := range args {
		val, err := strconv.ParseFloat(strval, 64)
		if err != nil {
			msg := "Cannot interpret %q as float. Try again.\n"
			return nil, fmt.Errorf(msg, strval)
		} else if val < 0.0 || val > 1.0 {
			msg := "Cannot interpret %q as float in [0, 1]. Try again.\n"
			return nil, fmt.Errorf(msg, strval)
		}
		parms[i] = val
	}
	return parms, nil
}
