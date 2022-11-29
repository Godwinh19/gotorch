package nn

/*
A loss function measure how good predictions are, it's used to adjust
the parameters of our network
*/
import (
	t "gotorch/torch/tensor"
)

type grad interface {
	loss() t.Tensor
	gradient() t.Tensor
}

type MSELoss struct {
	Predicted t.Tensor
	Actual    t.Tensor
}

func (l MSELoss) loss() t.Tensor {
	//numgo module for sum
	return t.Sum([]t.Tensor{t.Pow(t.Sub(l.Predicted, l.Actual), 2.0)}...)
}

func (l MSELoss) gradient() t.Tensor {
	return t.DotScalar(t.TensorOpsTensor(l.Predicted, l.Actual, "-"), 2.)
}

func Gradient(g grad) t.Tensor {
	return g.gradient()
}

func Loss(g grad) t.Tensor {
	return g.loss()
}