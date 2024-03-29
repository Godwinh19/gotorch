package nn

/*
Neural net will be made up of layers.
For each leayer we pass inputs forward
and propagate gradients backward.

Ex: inputs -> Linear -> Tanh -> Linear -> output

*/

import (
	t "github.com/Godwinh19/gotorch/torch/tensor"
)

type Layer struct {
	Params map[string]t.Tensor
	Grads  map[string]t.Tensor
}

type Linear struct {
	InputSize  int
	OutputSize int
	LLayer     Layer
	Activation Activation
	Inputs     t.Tensor // should we track each data for layer or make optional ?
}

func (l *Linear) InitializeParameters() {
	params := make(map[string]t.Tensor)
	params["w"] = *t.Rand([]int{l.InputSize, l.OutputSize})

	//params["b"] = t.Zeros(1, 1)
	params["b"] = *t.Zeros([]int{1, l.OutputSize})
	l.LLayer.Params = params
}

func (l *Linear) Forward(inputs t.Tensor) (outputs t.Tensor, weights map[string]t.Tensor) {
	/*
	   outputs = inputs @ w + b
	*/
	if l.LLayer.Params == nil {
		l.InitializeParameters()
	}
	params := l.LLayer.Params

	l.Inputs = inputs
	_dot, _ := t.Dot(inputs, params["w"])
	_out, _ := t.Sum(*_dot, params["b"])
	return *_out, params
}

func (l *Linear) Backward(grad t.Tensor) (gradients t.Tensor) {
	/*

			if y = f(x) and x = a * b + c
		        then dy/da = f'(x) * b
		        and dy/db = f'(x) * a
		        and dy/dc = f'(x)

		        if y = f(x) and x = a @ b + c
		        then dy/da = f'(x) @ b.T
		        and dy/db = a.T @ f'(x)
		        and dy/dc = f'(x)

	*/

	grads := make(map[string]t.Tensor)
	_sum, _ := t.Sum([]t.Tensor{grad}...)
	grads["b"] = *_sum
	_dot, _ := t.Dot(*l.Inputs.Transpose(), grad)
	grads["w"] = *_dot
	l.LLayer.Grads = grads

	weights := l.LLayer.Params["w"]
	_dot_w, _ := t.Dot(grad, *weights.Transpose())
	gradients = *_dot_w
	return
}

/*
Start Activation Layer
An activation layer just applies a function elementwise to its inputs
*/

// Activation Linear
type Activation struct {
	Name         string
	Inputs       t.Tensor
	forwardValue t.Tensor
}

func (a *Activation) isForwardValue() {
	if cap(a.forwardValue.Data) == 0 {
		panic("Compute the forward first")
	}
}

func (a *Activation) tanh(x t.Tensor) t.Tensor {
	activationValue := t.Tanh(x)
	a.forwardValue = activationValue
	return activationValue
}

func (a *Activation) tanh_prime(x t.Tensor) t.Tensor {
	a.isForwardValue()
	y := t.Pow(a.forwardValue, 2.)
	output := t.ScalarOpsTensor(1., y, "-")
	return *output
}

func (a *Activation) relu(x t.Tensor) t.Tensor {
	activationValue := t.ReLU(x)
	a.forwardValue = activationValue
	return activationValue
}

func (a *Activation) relu_prime(x t.Tensor) t.Tensor {
	a.isForwardValue()
	for i, value := range x.Data {
		if value < 0 {
			x.Data[i] = 0
		} else {
			x.Data[i] = 1
		}
	}
	return x
}

func (a *Activation) Forward(inputs t.Tensor) t.Tensor {
	a.Inputs = inputs
	// Currently work with tanh, relu
	if a.Name == "tanh" {
		return a.tanh(inputs)
	} else if a.Name == "relu" {
		return a.relu(inputs)
	} else {
		panic("Activation function not implemented")
	}
}

func (a *Activation) Backward(grad t.Tensor) t.Tensor {
	if a.Name == "tanh" {
		return *t.TensorOpsTensor(a.tanh_prime(a.Inputs), grad, "*")
	} else if a.Name == "relu" {
		return *t.TensorOpsTensor(a.relu_prime(a.Inputs), grad, "*")
	} else {
		panic("Activation function not implemented")
	}
}

func (a *Activation) IsExist() bool {
	return a.Name != ""
}

// End Activation Layer
