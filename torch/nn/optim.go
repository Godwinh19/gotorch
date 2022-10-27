package nn

/*
Optimizer is used to adjust networks parameters of network based on gradients
computing during backpropagation
*/

type SGD struct {
	Lr float64
}

func (sgd *SGD) step(net NeuralNet) {
}
