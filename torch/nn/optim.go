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
	
	params.Params["b"] = tensor.TensorOpsTensor(
	params.Params["b"], 
	tensor.DotScalar(params.Grads["b"], sgd.Lr),
		"-")
}
