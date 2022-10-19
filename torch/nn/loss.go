package loss

import t "nn/tensor"

/*
A loss function measure how good predictions are, it's used to adjust
the parameters of our network
*/

type Loss struct {
	predicted t.Tensor
	actual t.Tensor
}

func (l *Loss) MSELoss() float64 {
	//numgo module for sum
	sum((l.predicted - l.actual)**2)
}

func (l *Loss) MSEGrad() float64 {
	2 * (l.predicted - l.actual)
}

func sum(tensor t.Tensor) float64 {
	return 1.0
}