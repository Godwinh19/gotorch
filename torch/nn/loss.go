package nn

/*
A loss function measure how good predictions are, it's used to adjust
the parameters of our network
*/
import (
	t "gotorch/torch/tensor"
)

type Loss struct {
	Predicted t.Tensor
	Actual    t.Tensor
}

func (l *Loss) MSELoss() t.Tensor {
	//numgo module for sum
	return t.Sum([]t.Tensor{t.Pow(t.Sub(l.Predicted, l.Actual), 2.0)}...)
}

func (l *Loss) MSEGrad() t.Tensor {
	return t.DotScalar(t.Sub(l.Predicted, l.Actual), 2.)
}
