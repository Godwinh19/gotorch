package tensor

import (
	"errors"
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
	if len(dim) != len(t.Dim) {
		return nil, errors.New("Dimension mismatch")
	}
	newStride := make([]int, len(t.Stride))
	curStride := 1
	for i := len(t.Dim) - 1; i >= 0; i-- {
		if dim[i] != -1 && dim[i] != t.Dim[i] {
			return nil, errors.New("Dimension mismatch")
		}
		if dim[i] == -1 {
			dim[i] = t.Dim[i]
		}
		newStride[i] = curStride
		curStride *= dim[i]
	}
	if curStride != len(t.Data) {
		return nil, errors.New("Dimension mismatch")
	}
	return &Tensor{Dim: dim, Stride: newStride, Data: t.Data, RequiredGrad: t.RequiredGrad}, nil
}

func NewTensor(dim []int) *Tensor {
	size := 1
	stride := make([]int, len(dim))
	for i := len(dim) - 1; i >= 0; i-- {
		stride[i] = size
		size *= dim[i]
	}
	return &Tensor{
		Dim:    dim,
		Stride: stride,
		Data:   make([]float64, size),
	}
}
