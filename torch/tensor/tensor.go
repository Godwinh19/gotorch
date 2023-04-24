package tensor

import (
	"math"
	"math/rand"
)

type Tensor struct {
	Dim          []int
	Data         []float64
	RequiredGrad bool
}

func (t *Tensor) Shape() []int {
	return t.Dim
}

// Size returns the total number of elements in the tensor.
func (t *Tensor) Size() int {
	return Prod(t.Dim)
}

// Index returns the index of the element in the tensor data array
// corresponding to the given indices.
func (t *Tensor) Index(indices ...int) int {
	if len(indices) != len(t.Dim) {
		panic("Number of indices does not match tensor dimensions")
	}
	index := 0
	stride := 1
	for i := len(indices) - 1; i >= 0; i-- {
		if indices[i] < 0 || indices[i] >= t.Dim[i] {
			panic("Index out of range")
		}
		index += indices[i] * stride
		stride *= t.Dim[i]
	}
	return index
}

func (t *Tensor) ops(other interface{}, op string) *Tensor {
	switch other.(type) {
	case float64:
		return TensorOpsScalar(*t, other.(float64), op)
	case Tensor:
		return TensorOpsTensor(*t, other.(Tensor), op)
	case *Tensor:
		return TensorOpsTensor(*t, other.(Tensor), op)
	default:
		panic("Type error")
	}
}

func Zeros(shape []int) *Tensor {
	data := make([]float64, Prod(shape))
	return &Tensor{Dim: shape, Data: data}
}

func Ones(shape []int) *Tensor {
	data := make([]float64, Prod(shape))
	for i := range data {
		data[i] = 1
	}
	return &Tensor{Dim: shape, Data: data}
}

func Rand(shape []int) *Tensor {
	data := make([]float64, Prod(shape))
	for i := range data {
		data[i] = rand.Float64()
	}
	return &Tensor{Dim: shape, Data: data}
}

func (t *Tensor) Softmax(dim int) *Tensor {
	if dim < 0 || dim >= len(t.Dim) {
		panic("Invalid dimension")
	}
	_dim, _data := make([]int, len(t.Dim)), make([]float64, t.Size())
	output := NewTensor(_dim, _data, t.RequiredGrad)

	copy(output.Dim, t.Dim)

	stride := 1
	for i := dim + 1; i < len(t.Dim); i++ {
		stride *= t.Dim[i]
	}

	for i := 0; i < t.Size(); i += stride {
		max := t.Data[i]
		for j := i; j < i+stride; j++ {
			if t.Data[j] > max {
				max = t.Data[j]
			}
		}

		sum := 0.0
		for j := i; j < i+stride; j++ {
			output.Data[j] = math.Exp(t.Data[j] - max)
			sum += output.Data[j]
		}

		for j := i; j < i+stride; j++ {
			output.Data[j] /= sum
		}
	}

	return output
}
