package layers

/*
Neural net will be made up of layers.
For each leayer we pass inputs forward
and propagate gradients backward. 

Ex: inputs -> Linear -> Tanh -> Linear -> output

*/

import t "nn/tensor"


type Layer struct {

}

func (l *Layer) forward(inputs: t.Tensor) t.Tensor {
	return nil
} 

func (l *Layer) backward(grad: t.Tensor) t.Tensor {
	return nil
}