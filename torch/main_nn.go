package main

import (
	"fmt"
	"gotorch/torch/nn"
	t "gotorch/torch/tensor"
)

func main() {
	tensor := t.Rand(5, 4)
	linear_1 := nn.Linear{InputSize: 4, OutputSize: 5}
	linear_2 := nn.Linear{InputSize: 5, OutputSize: 3}

	one_forward := linear_1.Forward(tensor)
	out := linear_2.Forward(one_forward)

	fmt.Println(out.Shape())
	fmt.Println("Value after 2 forward", out)
	net := nn.NeuralNet{NLinear: []*nn.Linear{&linear_1, &linear_2}}
	output := net.Forward(tensor)
	net.Backward(output)
	optim := nn.SGD{Lr: 0.01}
	optim.Step(net)
}

//before test
