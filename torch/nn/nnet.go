package nn

import (
	t "gotorch/torch/tensor"
)

type NeuralNet struct {
	NLinear []*Linear
}

func (net *NeuralNet) Forward(inputs t.Tensor) t.Tensor {
	for _, layer := range net.NLinear {
		inputs = layer.Forward(inputs)
		inputs = layer.Activation.Forward(inputs)
	}
	return inputs
}

func (net *NeuralNet) Backward(grad t.Tensor) t.Tensor {
	for index := range net.NLinear {
		index++
		currentLayer := net.NLinear[len(net.NLinear)-index] // backward on reversed layers
		grad = currentLayer.Activation.Backward(grad)
		grad = currentLayer.Backward(grad)
	}

	return grad
}
