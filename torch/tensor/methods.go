package tensor

import (
	"errors"
)

func (t *Tensor) Transpose() *Tensor {
	shape := make([]int, len(t.Dim))
	copy(shape, t.Dim)
	for i := 0; i < len(shape)/2; i++ {
		shape[i], shape[len(shape)-1-i] = shape[len(shape)-1-i], shape[i]
	}
	newData := make([]float64, len(t.Data))
	idx := make([]int, len(shape))
	for i := 0; i < len(t.Data); i++ {
		n := i
		for j := len(idx) - 1; j >= 0; j-- {
			idx[j] = n % shape[j]
			n /= shape[j]
		}
		m := 0
		for j := len(idx) - 1; j >= 0; j-- {
			m = m*shape[j] + idx[j]
		}
		newData[i] = t.Data[m]
	}
	return &Tensor{Dim: shape, Data: newData, RequiredGrad: t.RequiredGrad}
}

func (t *Tensor) Reshape(shape []int) *Tensor {
	err := IsSameShape(shape, t.Dim)
	if err != nil {
		panic(err)
	}
	newTensor := &Tensor{
		Dim:          shape,
		Data:         make([]float64, t.Numel()),
		RequiredGrad: t.RequiredGrad,
	}

	for i := 0; i < t.Numel(); i++ {
		oldIdx := UnravelIndex(i, t.Dim)
		newIdx := UnravelIndex(i, newShape)
		newTensor.Set(newIdx, t.Get(oldIdx))
	}

	return newTensor
}

func (t *Tensor) UnravelIndex(idx int) []int {
	if idx < 0 || idx >= t.Size() {
		panic(errors.New("Index out of range"))
	}
	indices := make([]int, len(t.Dim))
	for i := len(t.Dim) - 1; i >= 0; i-- {
		indices[i] = idx % t.Dim[i]
		idx /= t.Dim[i]
	}
	return indices
}

func (t *Tensor) Set(indices []int, value float64) {
	idx := t.Index(indices)
	t.Data[idx] = value
}

func (t *Tensor) Get(indices []int) float64 {
	idx := t.Index(indices)
	return t.Data[idx]
}

func (t *Tensor) Index(indices []int) int {
	if len(indices) != len(t.Dim) {
		panic(errors.New("Number of indices does not match tensor dimension"))
	}
	var idx int
	for i := 0; i < len(t.Dim); i++ {
		if indices[i] < 0 || indices[i] >= t.Dim[i] {
			panic(errors.New("Index out of range"))
		}
		idx += indices[i] * t.Stride[i]
	}
	return idx
}

func (t *Tensor) Strides() []int {
	// Initialize the stride array to all ones
	stride := make([]int, len(t.Dim))
	for i := range stride {
		stride[i] = 1
	}

	// Compute the stride array based on the shape of the tensor
	for i := len(t.Dim) - 2; i >= 0; i-- {
		stride[i] = stride[i+1] * t.Dim[i+1]
	}

	return stride
}

func NewTensor(dim []int, data []float64, requiredGrad bool) *Tensor {
	return &Tensor{Dim: dim, Data: data, RequiredGrad: requiredGrad}
}
