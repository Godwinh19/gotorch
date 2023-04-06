package main

import (
	"fmt"
	"github.com/Godwinh19/gotorch/torch/nn"
	t "github.com/Godwinh19/gotorch/torch/tensor"
	"github.com/Godwinh19/gotorch/torch/utils"
	"path/filepath"
)

func main() {
	var data, _ = filepath.Abs("data/iris.csv")
	records := utils.ReadCsvFile(data)
	x, y, _ := utils.SplitXandY(records)
	fmt.Println(x.Shape(), y.Shape())
	training(x, y)
}

func training(x, y t.Tensor) {
	a1 := nn.Activation{Name: "tanh"}
	linear_1 := nn.Linear{InputSize: x.Shape()[1], OutputSize: 5, Activation: a1}
	a2 := nn.Activation{Name: "relu"}
	linear_2 := nn.Linear{InputSize: 5, OutputSize: 1, Activation: a2}

	var output, grad t.Tensor
	var params interface{}
	var currentLoss float64
	net := nn.NeuralNet{NLinear: []*nn.Linear{&linear_1, &linear_2}}
	lr := 0.0001
	optim := nn.SGD{Lr: lr}
	scheduler := nn.StepLRScheduler(lr, 10, 0.5)
	loss := nn.MSELoss{Actual: y}

	for i := 0; i < 100; i++ {

		// Zero gradients for every batch!
		optim.ZeroGradients(net)

		// Make predictions for this batch
		output, params = net.Forward(x)

		// Compute the loss and its gradients
		loss.Predicted = output
		grad = nn.Gradient(loss)

		currentLoss = float64(nn.Loss(loss).Data[0][0])
		net.Backward(grad)

		// Adjust learning weights
		optim.Step(net)
		optim.Lr = scheduler.Next()

		if i%5 == 0 {
			fmt.Println(currentLoss)
		}
	}
	xTest := utils.Random(x, 1)
	fmt.Println(xTest)
	y_test, _ := net.Forward(xTest)
	fmt.Println(y_test)
	fmt.Printf("\nParams for layers %v\n", params)
}
