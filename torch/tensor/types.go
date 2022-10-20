package tensor

import (
	"math/rand"
)

type Tensor struct {
	Rows, Cols int
	Data       [][]float64
	Requires_grad bool
}

func (t Tensor) Shape() []int {
	//assert that slice is homegenous
	AssertTensorFormat(t)
	t.Rows = cap(t.Data)
	t.Cols = len(t.Data[0])
	return []int{t.Rows, t.Cols}
}

func NewTensor(data [][]float64) Tensor {
	return Tensor{Data: data, Requires_grad: false}
}

func Rand(rows, cols int) Tensor {
	data := make([][]float64, rows)
	for i := 0; i<rows; i++ {
		data[i] = make([]float64, cols)
		for j := 0; j<cols; j++ {
			data[i][j] = rand.Float64()
		}
	}
	return Tensor{Data: data}
}