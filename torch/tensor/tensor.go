package tensor

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
)

type Tensor struct {
	Dim          []int
	Stride       []int
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

func NewTensor(dim []int) *Tensor {
	stride := Stride(dim)
	return &Tensor{
		Dim:    dim,
		Stride: stride,
		Data:   make([]float64, Prod(dim)),
	}
}

func (t *Tensor) Set(index []int, val float64) {
	idx := t.Index(index)
	t.Data[idx] = val
}

func (t *Tensor) Get(index []int) float64 {
	idx := t.Index(index)
	return t.Data[idx]
}

func (t *Tensor) Index(index []int) int {
	idx := 0
	for i := range t.Dim {
		if index[i] < 0 || index[i] >= t.Dim[i] {
			panic("Index out of range")
		}
		idx += index[i] * t.Stride[i]
	}
	return idx
}

func (t *Tensor) ops(other interface{}, op string) *Tensor {
	switch other.(type) {
	case float64:
		return TensorOpsScalar(*t, other.(float64), op)
	case Tensor:
		return TensorOpsTensor(*t, other.(Tensor), op)
	case *Tensor:
		return TensorOpsTensor(*t, *other.(*Tensor), op)
	default:
		panic("Type error")
	}
}

func (t *Tensor) Add(other interface{}) *Tensor {
	return t.ops(other, "+")
}

func (t *Tensor) Sub(other interface{}) *Tensor {
	return t.ops(other, "-")
}

func (t *Tensor) Mul(other interface{}) *Tensor {
	return t.ops(other, "*")
}

func (t *Tensor) Div(other interface{}) *Tensor {
	return t.ops(other, "/")
}

// Fill fills the tensor with the given value.
func (t *Tensor) Fill(value float64) {
	for i := range t.Data {
		t.Data[i] = value
	}
}

func Zeros(shape []int) *Tensor {
	return NewTensor(shape)
}

func Ones(shape []int) *Tensor {
	result := NewTensor(shape)
	for i := range result.Data {
		result.Data[i] = 1
	}
	return result
}

func Rand(shape []int) *Tensor {
	result := NewTensor(shape)
	for i := range result.Data {
		result.Data[i] = rand.Float64()
	}
	return result
}

// Softmax returns the softmax of the Tensor along the given dimension.
// @TODO check stride
func (t *Tensor) Softmax(dim int) (*Tensor, error) {
	if dim < 0 || dim >= len(t.Dim) {
		return nil, errors.New("invalid dimension")
	}

	// Calculate the output dimensions.
	outDim := make([]int, len(t.Dim))
	copy(outDim, t.Dim)
	outDim[dim] = 1

	// Calculate the output data.
	outData := make([]float64, Prod(outDim))
	var max float64
	fmt.Printf(" outt %v %v %v %v \n", t.Size(), outData, outDim, t.Dim)
	for i := 0; i < t.Size(); i += t.Dim[dim] {
		// Find the maximum value along the given dimension.
		max = t.Data[i]
		for j := 1; j < t.Dim[dim]; j++ {
			if t.Data[i+j] > max {
				max = t.Data[i+j]
			}
		}

		// Calculate the exponentials and sum along the given dimension.
		sum := 0.0
		for j := 0; j < t.Dim[dim]; j++ {
			fmt.Println("i, j", i, j)
			outData[i+j] = math.Exp(t.Data[i+j] - max)
			sum += outData[i+j]
		}

		// Normalize along the given dimension.
		for j := 0; j < t.Dim[dim]; j++ {
			outData[i+j] /= sum
		}
	}

	return &Tensor{Dim: outDim, Data: outData}, nil
}

// almostEquals returns true if the difference between a and b is less than or
// equal to the given tolerance.
func (t *Tensor) TensorAlmostEquals(b *Tensor) bool {
	tolerance := 0.0001
	for i, value := range t.Data {
		if math.Abs(value-b.Data[i]) > tolerance {
			return false
		}
	}
	return true
}
