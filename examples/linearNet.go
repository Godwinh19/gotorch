package main

import (
	"path/filepath"
	"fmt"
	"gotorch/torch/nn"
	t "gotorch/torch/tensor"
	"gotorch/torch/utils"
)

func main() {
	var data, _ = filepath.Abs("examples/data/iris.csv")
	records := utils.ReadCsvFile(data)
	x, y, _ := utils.SplitXandY(records)
    fmt.Println(x.Shape(), y.Shape())
	training(x, y)
}

func training(x, y t.Tensor) {
	a1 := nn.Activation{Name: "tanh"}
	linear_1 := nn.Linear{InputSize: int64(x.Shape()[1]), OutputSize: 5, Activation: a1}
	a2 := nn.Activation{Name: "relu"}
	linear_2 := nn.Linear{InputSize: 5, OutputSize: 1, Activation: a2}

	var output, grad t.Tensor
	var currentLoss float64
	net := nn.NeuralNet{NLinear: []*nn.Linear{&linear_1, &linear_2}}
	optim := nn.SGD{Lr: 0.00001}
	loss := nn.MSELoss{Actual: y}

	for i := 0; i < 1; i++ {
		//for each epoch
		for i := 0; i < 100; i++ {
			
			// Zero gradients for every batch!
			optim.ZeroGradients(net)
			
			// Make predictions for this batch
			output = net.Forward(x)

			// Compute the loss and its gradients
			loss.Predicted = output
			grad = nn.Gradient(loss)

			currentLoss = float64(nn.Loss(loss).Data[0][0])
			net.Backward(grad)

			// Adjust learning weights
			optim.Step(net)
			
			if i%5 == 0 {
				fmt.Println(currentLoss)
			}
		}
	}
	x_test := utils.Random(x, 1)
	fmt.Println(x_test)
	fmt.Println(net.Forward(x_test))
}
