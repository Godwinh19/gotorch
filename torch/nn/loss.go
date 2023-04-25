package nn

/*
A loss function measure how good predictions are, it's used to adjust
the parameters of our network
*/
import (
	t "github.com/Godwinh19/gotorch/torch/tensor"
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
	_out, _ := t.Sum([]t.Tensor{t.Pow(*t.TensorOpsTensor(l.Predicted, l.Actual, "-"), 2.0)}...)
	_out.Data[0] = _out.Data[0] / float64(len(l.Predicted.Data))
	return *_out
}

func (l MSELoss) gradient() t.Tensor {
	_out := t.TensorOpsScalar(*t.TensorOpsTensor(l.Predicted, l.Actual, "-"), 2., "*")
	_out.Data[0] = _out.Data[0] / float64(len(l.Predicted.Data))
	return *_out
}

func Gradient(g grad) t.Tensor {
	return g.gradient()
}

func Loss(g grad) t.Tensor {
	return g.loss()
}
