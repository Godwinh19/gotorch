package nn

import (
	t "gotorch/torch/tensor"
)

type NeuralNet struct {
	NLinear []*Linear
}

func (net *NeuralNet) Forward(inputs t.Tensor) (t.Tensor, interface{}) {
	//This function compute the forward and return the layer output and all weights
	weights := make(map[int]map[string]t.Tensor, len(net.NLinear))
	for idx, layer := range net.NLinear {
		inputs, weights[idx] = layer.Forward(inputs)
		if layer.Activation.IsExist() {
			inputs = layer.Activation.Forward(inputs)
		}
	}
	return inputs, weights
}

func (net *NeuralNet) Backward(grad t.Tensor) t.Tensor {
	for index := range net.NLinear {
		index++
		currentLayer := net.NLinear[len(net.NLinear)-index] // backward on reversed layers
		if currentLayer.Activation.IsExist() {
			grad = currentLayer.Activation.Backward(grad)
		}
		grad = currentLayer.Backward(grad)
	}

	return grad
}
