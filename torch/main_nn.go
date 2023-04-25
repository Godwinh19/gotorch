package main

import (
	"fmt"
	"github.com/Godwinh19/gotorch/torch/nn"
	t "github.com/Godwinh19/gotorch/torch/tensor"
)

func main() {
	tensor := *t.Rand([]int{5, 4})
	linear_1 := nn.Linear{InputSize: 4, OutputSize: 5}
	linear_2 := nn.Linear{InputSize: 5, OutputSize: 3}

	one_forward, _ := linear_1.Forward(tensor)
	out, _ := linear_2.Forward(one_forward)

	fmt.Println(out.Shape())
	fmt.Println("Value after 2 forward", out)
	net := nn.NeuralNet{NLinear: []*nn.Linear{&linear_1, &linear_2}}
	output, _ := net.Forward(tensor)
	net.Backward(output)
	optim := nn.SGD{Lr: 0.01}
	optim.Step(net)
}

//before test
