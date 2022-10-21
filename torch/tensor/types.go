package tensor

import (
	"math/rand"
)

type Tensor struct {
	Rows, Cols   int
	Data         [][]float64
	RequiresGrad bool
}

func (t Tensor) Shape() []int {
	//assert that slice is homogeneous
	AssertTensorFormat(t)
	t.Rows = cap(t.Data)
	t.Cols = len(t.Data[0])
	return []int{t.Rows, t.Cols}
}

func (t Tensor) Transpose() Tensor {
	output := Zeros(t.Rows, t.Cols).Data

	for i := 0; i < t.Rows; i++ {
		for j := 0; j < t.Cols; j++ {
			output[i][j] = t.Data[j][i]
		}
	}

	return Tensor{Data: output, Rows: t.Cols, Cols: t.Rows}
}

func NewTensor(data [][]float64) Tensor {
	return Tensor{Data: data, RequiresGrad: false}
}

func Rand(rows, cols int) Tensor {
	data := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		data[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			data[i][j] = rand.Float64()
		}
	}
	return Tensor{Data: data, RequiresGrad: false, Cols: cols, Rows: rows}
}

func Zeros(rows, cols int) Tensor {
	data := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		data[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			data[i][j] = 0
		}
	}
	return Tensor{Data: data, RequiresGrad: false, Cols: cols, Rows: rows}
}
