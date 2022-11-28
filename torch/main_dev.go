package main

import (
	"fmt"
	"gotorch/torch/nn"
	t "gotorch/torch/tensor"
)

func main() {
	x1 := t.Rand(5, 4)
	linear_1 := nn.Linear{InputSize: 4, OutputSize: 5}
	linear_2 := nn.Linear{InputSize: 5, OutputSize: 3}
	linear_3 := nn.Linear{InputSize: 3, OutputSize: 1}

	net := nn.NeuralNet{NLinear: []*nn.Linear{&linear_1, &linear_2, &linear_3}}
	optim := nn.SGD{Lr: 0.01}
	for i := 0; i < 10; i++ {
		output := net.Forward(x1)
		net.Backward(output)
		optim.Step(net)
	}
	// Next add activation for each forward operation
	fmt.Println(net.NLinear[2].LLayer)
	fmt.Println("")
}

