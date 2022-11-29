package nn

import (
	"gotorch/torch/tensor"
)

/*
Optimizer is used to adjust networks parameters of network based on gradients
computing during backpropagation
*/

type SGD struct {
	Lr float64
}

func (sgd *SGD) ZeroGradients(net NeuralNet) {
	for _, lin := range net.NLinear {
		lin.LLayer.Grads = nil
	}
}

func (sgd *SGD) Step(net NeuralNet) {
	for _, lin := range net.NLinear {
		sgd.updateParameters(lin.LLayer)
	}
}

func (sgd *SGD) updateParameters(params Layer) {
	params.Params["w"] = tensor.TensorOpsTensor(
		params.Params["w"], 
		tensor.DotScalar(params.Grads["w"], sgd.Lr),
		 "-")

	//params.Grads["b"] is a scalar then
	params.Params["b"] = tensor.TensorOpsScalar(
	params.Params["b"], 
	tensor.DotScalar(params.Grads["b"], sgd.Lr).Data[0][0],
		"-")
}
