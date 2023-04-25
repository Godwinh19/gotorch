package tensor

import (
	"errors"
	"fmt"
	"math"
)

// Mul performs element-wise multiplication between two tensors of the same shape
func Mul(a, b Tensor) (*Tensor, error) {
	if !IsSameShape(a.Dim, b.Dim) {
		return nil, errors.New("tensors must have the same shape")
	}

	result := NewTensor(a.Dim)
	for i := 0; i < len(a.Data); i++ {
		result.Data[i] = a.Data[i] * b.Data[i]
	}
	return result, nil
}

// Dot performs matrix multiplication between two tensors of compatible shapes
func Dot(a, b Tensor) (*Tensor, error) {
	if len(a.Dim) != 2 || len(b.Dim) != 2 || a.Dim[1] != b.Dim[0] {
		return nil, errors.New("tensors must be 2D and have compatible dimensions for matrix multiplication")
	}
	if a.Dim[1] != b.Dim[0] {
		return nil, errors.New("matrices are not compatible for multiplication")
	}
	resultDim := []int{a.Dim[0], b.Dim[1]}
	result := NewTensor(resultDim)
	for i := 0; i < a.Dim[0]; i++ {
		for j := 0; j < b.Dim[1]; j++ {
			for k := 0; k < a.Dim[1]; k++ {
				result.Data[i*b.Dim[1]+j] += a.Data[i*a.Dim[1]+k] * b.Data[k*b.Dim[1]+j]
				//result.Data[i*result.Stride[0]+j] += a.Data[i*a.Stride[0]+k] * b.Data[k*b.Stride[0]+j]
			}
		}
	}
	return result, nil
}

// Sum performs element-wise addition between two tensors a and b.
// If b has smaller dimensions than a, b is broadcast to match the shape of a.
// It returns a new tensor with the same dimensions as a and b, where each element
// is the sum of the corresponding elements in a and b.
func Sum(params ...Tensor) (*Tensor, error) {
	if len(params) == 1 {
		a := params[0]
		sum := 0.0
		for _, value := range a.Data {
			sum += value
		}
		return &Tensor{Dim: []int{1}, Data: []float64{sum}, RequiredGrad: a.RequiredGrad}, nil
	} else if len(params) == 2 {
		a, b := &params[0], &params[1] // the second parameter is used for broadcasting
		if !IsSameShape(a.Dim, b.Dim) {
			if !broadcastable(a.Dim, b.Dim) {
				return nil, errors.New("tensors are not broadcastable for element-wise addition")
			}
			b, _ = b.Broadcast(a.Dim)
		}
		c := NewTensor(a.Dim)
		for i := range a.Data {
			c.Data[i] = a.Data[i] + b.Data[i]
		}
		return c, nil
	} else {
		fmt.Errorf("Dimensional operation not supported")
		return &Tensor{}, errors.New("Dimensional operation not supported")
	}
}

// broadcastable checks if the dimensions of two tensors are broadcastable.
func broadcastable(a, b []int) bool {
	if len(a) < len(b) {
		a, b = b, a
	}
	for i := 1; i <= len(b); i++ {
		if a[len(a)-i] != b[len(b)-i] && a[len(a)-i] != 1 && b[len(b)-i] != 1 {
			return false
		}
	}
	return true
}

// Broadcast returns a new tensor with the same data as t, but with dimensions d,
// obtained by broadcasting t according to numpy broadcasting rules.
// See https://numpy.org/doc/stable/user/basics.broadcasting.html for more information.
// minimal implementation
func (t *Tensor) Broadcast(d []int) (*Tensor, error) {
	if len(t.Dim) > len(d) {
		return nil, errors.New("cannot broadcast to fewer dimensions")
	}
	out := Ones(d)

	for i, _ := range out.Data {
		out.Data[i] = t.Data[i%Prod(t.Dim)]
	}

	return out, nil
}

func TensorOpsTensor(a Tensor, b Tensor, op string) *Tensor {
	if !IsSameShape(a.Dim, b.Dim) {
		return TensorOpsTensorWithBroadcasting(a, b, op)
	}
	output := Zeros(a.Dim)
	for i, value := range a.Data {
		output.Data[i] = linearOperation(value, b.Data[i], op)
	}
	return output
}

// TensorOpsTensorWithBroadcasting perform operation while taking account broadcast
func TensorOpsTensorWithBroadcasting(a Tensor, b Tensor, ops string) *Tensor {
	return &Tensor{}
}

func TensorOpsScalar(a Tensor, b float64, op string) *Tensor {
	output := Zeros(a.Dim)
	for i, value := range a.Data {
		output.Data[i] = linearOperation(value, b, op)
	}
	return output
}

func ScalarOpsTensor(a float64, b Tensor, op string) *Tensor {
	output := Zeros(b.Dim)
	for i, value := range b.Data {
		output.Data[i] = linearOperation(a, value, op)
	}
	return output
}

func linearOperation(a float64, b float64, op string) float64 {
	if op == "minus" || op == "-" {
		return a - b
	} else if op == "plus" || op == "+" {
		return a + b
	} else if op == "mul" || op == "*" {
		return a * b
	} else if op == "div" || op == "/" {
		return a / b
	} else {
		panic("Error operations")
	}
}

func Tanh(a Tensor) Tensor {
	for i, value := range a.Data {
		a.Data[i] = math.Tanh(value)
	}
	return a
}

func ReLU(a Tensor) Tensor {
	for i, value := range a.Data {
		if value < 0 {
			a.Data[i] = 0
		}
	}
	return a
}

func Pow(a Tensor, pow float64) Tensor {
	for i, value := range a.Data {
		a.Data[i] = math.Pow(value, pow)
	}
	return a
}
