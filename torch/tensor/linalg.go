package tensor

import (
	"fmt"
	"math"
)

func Mul(a, b Tensor) *Tensor {
	err := IsSameShape(a.Dim, b.Dim)
	if err != nil {
		panic(err)
	}
	output := Zeros(a.Dim)
	for i := range a.Data {
		output.Data[i] = a.Data[i] * b.Data[i]
	}
	return output
}

func Dot(a, b Tensor) *Tensor {
	// check that dimensions are compatible for matrix multiplication
	if a.Dim[len(a.Dim)-1] != b.Dim[len(b.Dim)-2] {
		panic("Incompatible dimensions for matrix multiplication")
	}

	// allocate space for output tensor
	outDim := []int{a.Dim[0], b.Dim[1]}
	out := Zeros(outDim)
	out.RequiredGrad = a.RequiredGrad

	// flatten the tensors
	aFlat := a.Reshape([]int{a.Dim[0], -1})
	bFlat := b.Reshape([]int{-1, b.Dim[1]})

	// perform matrix multiplication
	for i := 0; i < a.Dim[0]; i++ {
		for j := 0; j < b.Dim[1]; j++ {
			sum := 0.0
			for k := 0; k < a.Dim[len(a.Dim)-1]; k++ {
				sum += aFlat.Data[i*a.Dim[len(a.Dim)-1]+k] * bFlat.Data[k*b.Dim[1]+j]
			}
			out.Data[i*out.Dim[1]+j] = sum
		}
	}

	// restore the original shape of the output tensor
	out = out.Reshape(outDim)

	return out
}

func Sum(params ...Tensor) *Tensor {
	if len(params) == 1 {
		a := params[0]
		sum := 0.0
		for _, value := range a.Data {
			sum += value
		}
		return &Tensor{Dim: []int{1}, Data: []float64{sum}, RequiredGrad: a.RequiredGrad}
	} else if len(params) == 2 {
		a, b := params[0], params[1] // the second parameter is used for broadcasting
		aShape, bShape := a.Shape(), b.Shape()
		// Ensure that b is a 1D tensor with the same number of columns as a
		if len(bShape) != 2 || bShape[0] != 1 || bShape[1] != aShape[1] {
			panic("Invalid shape for second tensor")
		}

		// Create a new Tensor to hold the result
		output := Zeros(aShape)
		aStr, bStr := a.Strides(), b.Strides()

		// Compute the element-wise sum between a and b
		for i := 0; i < aShape[0]; i++ {
			for j := 0; j < aShape[1]; j++ {
				output.Data[i*aStr[0]+j*a.Strides()[1]] = a.Data[i*aStr[0]+j*aStr[1]] + b.Data[j*bStr[1]]
			}
		}

		return output
	} else {
		fmt.Errorf("Dimensional operation not supported")
		return &Tensor{}
	}
}

func TensorOpsTensor(a Tensor, b Tensor, op string) *Tensor {
	err := IsSameShape(a.Dim, b.Dim)
	if err != nil {
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
