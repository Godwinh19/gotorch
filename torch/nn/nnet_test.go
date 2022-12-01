package nn

import (
	"testing"
	"gotorch/torch/tensor"
)

func TestNet(t *testing.T) {
	x1 := tensor.Rand(100, 12)
	a1 := Activation{Name: "tanh"}
	linear_1 := Linear{InputSize: 12, OutputSize: 5, Activation: a1}
	a2 := Activation{Name: "tanh"}
	linear_2 := Linear{InputSize: 5, OutputSize: 3, Activation: a2}
	a3 := Activation{Name: "relu"}
	linear_3 := Linear{InputSize: 3, OutputSize: 1, Activation: a3}

	net := NeuralNet{NLinear: []*Linear{&linear_1, &linear_2, &linear_3}}
	_ = net.Forward(x1)
	_ = net.Backward(x1)

}