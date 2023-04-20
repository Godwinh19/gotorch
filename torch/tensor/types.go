package tensor

import (
	"math/rand"
)

type ITensor interface {
	Shape()
	Rand()
}

type Tensor struct {
	Rows, Cols   int
	Data         [][]float64
	RequiresGrad bool
}

type NTensor struct {
	Dim          []int
	Data         []float64
	RequiredGrad bool
}

func (t Tensor) Shape() []int {
	//assert that slice is homogeneous
	_ = AssertTensorFormat(t)
	t.Rows = cap(t.Data)
	t.Cols = len(t.Data[0])
	return []int{t.Rows, t.Cols}
}

func (t *NTensor) Shape() []int {
	return t.Dim
}

func (t Tensor) Transpose() Tensor {
	output := Zeros(t.Cols, t.Rows).Data

	for i := 0; i < t.Cols; i++ {
		for j := 0; j < t.Rows; j++ {
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

func (t *NTensor) Rand() NTensor {
	return NTensor{}
}

func Zeros(rows, cols int) Tensor {
	data := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		data[i] = make([]float64, cols)
	}
	return Tensor{Data: data, RequiresGrad: false, Cols: cols, Rows: rows}
}

func (t *Tensor) Slice(startRow, endRow, startCol, endCol int) *Tensor {
	if startRow < 0 || startRow >= t.Rows || endRow < startRow || endRow > t.Rows ||
		startCol < 0 || startCol >= t.Cols || endCol < startCol || endCol > t.Cols {
		return nil
	}

	rows := endRow - startRow
	cols := endCol - startCol

	newData := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		newData[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			newData[i][j] = t.Data[i+startRow][j+startCol]
		}
	}

	return &Tensor{
		Rows:         rows,
		Cols:         cols,
		Data:         newData,
		RequiresGrad: t.RequiresGrad,
	}
}
