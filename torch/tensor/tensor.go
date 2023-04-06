package tensor

import (
	"math"
	//"reflect"
)

func BuildTensor(dimension int, numAxes int) interface{} {
	if numAxes == 1 {
		return make([]float64, dimension)
	} else {
		data := make([]interface{}, dimension)
		for i := 0; i < dimension; i++ {
			data[i] = BuildTensor(dimension, numAxes-1)
		}
		return data
	}

	//return Tensor{Data: data.(make(int, numAxes))}
}

func (t *Tensor) Reshape(shape []int) Tensor {
	// Check if the number of elements in the tensor
	// is equal to the number of elements in the new shape

	size := 1
	for _, s := range shape {
		size *= s
	}

	if len(t.Data) != size {
		panic("Total size of new array must be unchanged")
	}

	// t.Data = reflect.ValueOf(t.Data).Slice(0, size).Interface().([]float64)
	return *t
}

func Softmax(tensor Tensor) Tensor {
	// TODO: tensor format not yet stable
	// Assume that tensor are 1-vector logits
	//tensor = tensor.Reshape(1, -1)

	sum := 0.0
	for _, value := range tensor.Data[0] {
		sum += math.Exp(value)
	}

	for i, value := range tensor.Data[0] {
		tensor.Data[0][i] = math.Exp(value) / sum
	}

	return tensor
}
