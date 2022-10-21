package nn

/*
Neural net will be made up of layers.
For each leayer we pass inputs forward
and propagate gradients backward.

Ex: inputs -> Linear -> Tanh -> Linear -> output

*/

import (
	t "gotorch/torch/tensor"
)

type Layer struct {
	Params map[string]t.Tensor
	Grads  map[string]t.Tensor
}

type Linear struct {
	InputSize  int64
	OutputSize int64
	LLayer     Layer
	Inputs     t.Tensor
}

func (l *Linear) Forward(inputs t.Tensor) (outputs t.Tensor) {
	/*
	   outputs = inputs @ w + b
	*/

	params := make(map[string]t.Tensor)
	params["w"] = t.Rand(int(l.InputSize), int(l.OutputSize))

	params["b"] = t.Zeros(1, int(l.OutputSize))

	l.LLayer.Params = params
	l.Inputs = inputs
	outputs = t.Sum([]t.Tensor{t.Dot(inputs, params["w"]), params["b"]}...)
	return
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
	grads["b"] = t.Sum([]t.Tensor{grad}...)
	grads["w"] = t.Dot(l.Inputs.Transpose(), grad)
	l.LLayer.Grads = grads

	gradients = t.Dot(grad, l.LLayer.Params["w"].Transpose())
	return
}

/*
Start Activation Layer
An activation layer just applies a function elementwise to its inputs
*/

// Linear
type Activation struct {
	LLayer Layer
}

func (a *Activation) tanh(x t.Tensor) t.Tensor {
	return t.Tanh(x)
}

func (a *Activation) tanh_prime(x t.Tensor) t.Tensor {
	y := a.tanh(x)
	t.Pow(y, 2.)
	output := t.ScalarMinusTensor(y, 1.)
	return output
}

// End Activation Layer

type Forward interface {
	forward()
}

type Backward interface {
	backward()
}
