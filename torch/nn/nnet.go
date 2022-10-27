package nn

import t "gotorch/torch/tensor"

type NeuralNet struct {
	NLinear []Linear
}

func (net *NeuralNet) Forward(inputs t.Tensor) t.Tensor {
	for _, layer := range net.NLinear {
		inputs = layer.Forward(inputs)
	}
	return inputs
}

func (net *NeuralNet) Backward(grad t.Tensor) t.Tensor {
	var currentLayer Linear
	for index := range net.NLinear {
		currentLayer = net.NLinear[len(net.NLinear)-index] // backward on reversed layers
		grad = currentLayer.Backward(grad)
	}

	return grad
}
