package tensor

import (
	"errors"
	"fmt"
)

func (t *Tensor) Transpose() *Tensor {
	if len(t.Dim) < 2 {
		return t
	}
	newDim := make([]int, len(t.Dim))
	newStride := make([]int, len(t.Stride))
	for i := range newDim {
		newDim[i] = t.Dim[len(t.Dim)-i-1]
		newStride[i] = t.Stride[len(t.Stride)-i-1]
	}
	return &Tensor{Dim: newDim, Stride: newStride, Data: t.Data, RequiredGrad: t.RequiredGrad}
}

func (t *Tensor) Reshape(dim []int) (*Tensor, error) {
	// Compute the total number of elements in the new shape
	if Prod(dim) != Prod(t.Dim) {
		return nil, errors.New("Dimension mismatch")
	}

	// Compute the new stride values
	newStride := make([]int, len(dim))
	curStride := 1
	for i := len(dim) - 1; i >= 0; i-- {
		newStride[i] = curStride
		curStride *= dim[i]
	}

	// Check if the new stride values are valid
	if curStride != len(t.Data) {
		return nil, errors.New("Dimension mismatch")
	}

	// Create the new tensor with the updated shape and stride
	return &Tensor{Dim: dim, Stride: newStride, Data: t.Data, RequiredGrad: t.RequiredGrad}, nil
}

func (t *Tensor) GetStride() []int {
	return Stride(t.Dim)
}

func (t *Tensor) Print() {
	fmt.Print("tensor(")
	if len(t.Dim) == 0 {
		fmt.Println("[]")
		return
	}
	printHelper(t.Data, t.Dim, t.Stride, 0, make([]int, len(t.Dim)))
	fmt.Printf("Shape: %v and Stride: %v )\n", t.Dim, t.Stride)
}

func printHelper(data []float64, dim []int, stride []int, offset int, indices []int) {
	if len(dim) == 1 {
		fmt.Print("[ ")
		for i := 0; i < dim[0]; i++ {
			fmt.Printf("%v", data[offset+i*stride[0]])
			if i != dim[0]-1 {
				fmt.Print(",")
			}
		}
		fmt.Println("],")
		return
	}
	fmt.Print("[")
	for i := 0; i < dim[0]; i++ {
		indices[0] = i
		printHelper(data, dim[1:], stride[1:], offset+i*stride[0], indices[1:])
	}
	fmt.Println("]")
}
