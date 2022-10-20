package nn

/*
Neural net will be made up of layers.
For each leayer we pass inputs forward
and propagate gradients backward.

Ex: inputs -> Linear -> Tanh -> Linear -> output

*/

import (
	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	Params map[string]Tensor
	Grad map[string]Tensor
}

type Linear struct {
	Input_size  int64
	Output_size int64
	LLayer      Layer
	Inputs      Tensor
}

func (l *Linear) forward(inputs Tensor) (outputs Tensor) {
	/*
        outputs = inputs @ w + b
	*/

	params := make(map[string]Tensor)
	params["w"] = Tensor{Data: inputs.Data.Mul(l.Input_size, l.Output_size),
		Requires_grad: true,
		Shape: []int64{l.Input_size, l.Output_size},
	}
	params["b"] = Tensor{Data: inputs.Data.Mul(l.Output_size),
		Requires_grad: true,
		Shape: []int64{1, l.Output_size},
	}

	l.LLayer.Params = params
	l.Inputs = inputs
	regression := mat.Dot(inputs.Data, params["w"].Data) + params["b"].Data
	outputs := Tensor{Data: regression, Requires_grad: true, 
		Shape: []int64{l.Input_size, l.Output_size}}
	return 
}

func (l *Linear) backward(grad Tensor) (gradients Tensor) {
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

	grads := make(map[string]Tensor)
	grads["b"] = mat.Sum(grad.Data)
	grads["w"] = mat.Dot(l.Inputs.Data.T, grad.Data)
	l.Grads = grads

	gradients := mat.Dot(grad, l.LLayer.Params["w"].Data.T)
	return
}

/*
Start Activation Layer
An activation layer just applies a function elementwise to its inputs
*/

//Linear
type Activation struct {
	LLayer Layer
}

func (a *Activation) tanh(x Tensor) Tensor {
	return Tensor{Data: mat.Tanh(x.Data),
	Requires_grad: true,
	Shape: []int64{x.Data.Rows(), 1},
    }
}

func (a *Activation) tanh_prime(x Tensor) Tensor {
	y := a.tanh(x)
	return Tensor{Data: 1 - mat.Pow(y, 2),
	Requires_grad: true,
	Shape: []int64{x.Data.Rows(), 1},
    }
}


// End Activation Layer


type Forward interface {
	forward()
}

type Backward interface {
	backward()
}
