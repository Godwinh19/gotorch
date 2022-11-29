package main

import (
	"fmt"
	"gotorch/torch/nn"
	t "gotorch/torch/tensor"
)

func main() {
	x1 := t.Rand(500, 4)
	a1 := nn.Activation{Name: "tanh"}
	linear_1 := nn.Linear{InputSize: 4, OutputSize: 5, Activation: a1}
	a2 := nn.Activation{Name: "tanh"}
	linear_2 := nn.Linear{InputSize: 5, OutputSize: 3, Activation: a2}
	a3 := nn.Activation{Name: "tanh"}
	linear_3 := nn.Linear{InputSize: 3, OutputSize: 1, Activation: a3}

	var output, grad t.Tensor
	var currentLoss float64
	net := nn.NeuralNet{NLinear: []*nn.Linear{&linear_1, &linear_2, &linear_3}}
	optim := nn.SGD{Lr: 0.01}
	loss := nn.MSELoss{Actual: t.Rand(500,1)}
	//fmt.Println(loss.Actual)
	for i := 0; i < 1; i++ {
		//for each epoch
		for i := 0; i < 10; i++ {
			// for each batch, next we'll create data batch
			// compute loss for each batch

			// Zero gradients for every batch!
			optim.ZeroGradients(net)
			
			// Make predictions for this batch
			output = net.Forward(x1)

			// Compute the loss and its gradients
			loss.Predicted = output
			grad = nn.Gradient(loss)

			currentLoss = float64(nn.Loss(loss).Data[0][0])
			net.Backward(grad)

			// Adjust learning weights
			optim.Step(net)
			fmt.Println(currentLoss)
		}
	}
	// Next add activation for each forward operation
	//fmt.Println(net.NLinear[2].LLayer)
	fmt.Println("")
}
