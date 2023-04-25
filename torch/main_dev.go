//move to test later

package main

import (
	"fmt"
	"github.com/Godwinh19/gotorch/torch/nn"
	t "github.com/Godwinh19/gotorch/torch/tensor"
	"github.com/Godwinh19/gotorch/torch/utils"
	"reflect"
)

func buildTensor() {

	tensor := t.NewTensor([]int{2, 3})
	/// TODO: investigate on buiding multi dimensional tensors
	// and convert to float64
	//fmt.Printf("%v\n",tensor)
	t2 := reflect.ValueOf(tensor)

	for i := 0; i < t2.Len(); i++ {
		fmt.Printf("%v\n", t2.Index(i).Interface())
	}

	/*
			var data, _ = filepath.Abs("examples/data/iris.csv")
			records := utils.ReadCsvFile(data)
			x, y, _ := utils.SplitXandY(records)
		    fmt.Println(x.Shape(), y.Shape())
			training(x, y)
	*/
}

func training(x, y t.Tensor) {
	//x1 := t.Rand(100, 12)
	a1 := nn.Activation{Name: "tanh"}
	linear_1 := nn.Linear{InputSize: x.Shape()[1], OutputSize: 5, Activation: a1}
	a2 := nn.Activation{Name: "tanh"}
	linear_2 := nn.Linear{InputSize: 5, OutputSize: 3, Activation: a2}
	a3 := nn.Activation{Name: "relu"}
	linear_3 := nn.Linear{InputSize: 3, OutputSize: 1, Activation: a3}

	var output, grad t.Tensor
	var currentLoss float64
	net := nn.NeuralNet{NLinear: []*nn.Linear{&linear_1, &linear_2, &linear_3}}
	lr := 0.00001
	optim := nn.SGD{Lr: lr}
	scheduler := nn.StepLRScheduler(lr, 10, 0.5)
	loss := nn.MSELoss{Actual: y}

	for i := 0; i < 1; i++ {
		//for each epoch
		for i := 0; i < 100; i++ {
			// for each batch, next we'll create data batch
			// compute loss for each batch

			// Zero gradients for every batch!
			optim.ZeroGradients(net)

			// Make predictions for this batch
			output, _ = net.Forward(x)

			// Compute the loss and its gradients
			loss.Predicted = output
			grad = nn.Gradient(loss)

			currentLoss = float64(nn.Loss(loss).Data[0])
			net.Backward(grad)

			// Adjust learning weights
			optim.Step(net)
			// the next lr will be the lr returned by scheduler
			optim.Lr = scheduler.Next()

			if i%1 == 0 {
				fmt.Println(currentLoss)
			}
		}
	}
	// Next add softmax function
	//fmt.Println(net.NLinear[2].LLayer)
	x_test := utils.Random(x, 4)
	fmt.Println(x_test)
	fmt.Println((net.Forward(x_test)))
}
