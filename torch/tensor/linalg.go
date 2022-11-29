package tensor

import (
	"fmt"
	"math"
)

func Dot(a, b Tensor) Tensor {
	err := a.SameTensorRowColShape(b)
	if err != nil {
		panic(err.Error())
	}
	output := Zeros(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			j_vect := VerticalVector(b, j)
			output.Data[i][j] = dot(a.Data[i], j_vect)
		}
	}
	return output
}

func DotScalar(a Tensor, scalar float64) Tensor {
	output := Zeros(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			output.Data[i][j] = a.Data[i][j] * scalar
		}
	}
	return output
}

func dot(a, b []float64) float64 {
	la := len(a) //we assume a and b are same dimension
	sum := 0.0
	for i := 0; i < la; i++ {
		sum += a[i] * b[i]
	}

	return sum
}

func Sum(params ...Tensor) Tensor {
	if len(params) == 1 {
		a := params[0]
		sum := 0.0
		for i := 0; i < a.Cols; i++ {
			sum += a.Data[0][i]
		}
		return Tensor{Data: [][]float64{{sum}}, Rows: 1, Cols: 1}
	} else if len(params) == 2 {
		a, b := params[0], params[1] // the second parameter is used for broadcasting
		output := Zeros(a.Rows, b.Cols)
		for i := 0; i < a.Rows; i++ {
			for j := 0; j < b.Cols; j++ {
				output.Data[i][j] = a.Data[i][j] + b.Data[0][j] //b is 1-d data
			}
		}
		return output
	} else {
		fmt.Errorf("Dimensional operation not supported")
		return Tensor{}
	}
}

func Sub(a, b Tensor) Tensor {
	err := a.SameTensorShape(b)
	if err != nil {
		panic(err.Error())
	}
	output := Zeros(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			output.Data[i][j] = a.Data[i][j] - b.Data[i][j]
		}
	}
	return output
}

func AddScalar(a Tensor, scalar float64) Tensor {
	output := Zeros(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			output.Data[i][j] = a.Data[i][j] + scalar
		}
	}
	return output
}

func ScalarMinusTensor(a Tensor, scalar float64) Tensor {
	output := Zeros(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			output.Data[i][j] = scalar - a.Data[i][j]
		}
	}
	return output
}

func TensorOpsTensor(a Tensor, b Tensor, ops string) Tensor {
	err := a.SameTensorShape(b)
	if err != nil {
		return TensorOpsTensorWithBroadcasting(a, b, ops)
	}
	output := Zeros(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			output.Data[i][j] = linearOperation(a.Data[i][j], b.Data[i][j], ops)
		}
	}
	return output
}

func TensorOpsTensorWithBroadcasting(a Tensor, b Tensor, ops string) Tensor {
	output := Zeros(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			output.Data[i][j] = linearOperation(a.Data[i][j], b.Data[0][j], ops)
		}
	}
	return output
}

func TensorOpsScalar(a Tensor, b float64, ops string) Tensor {
	output := Zeros(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			output.Data[i][j] = linearOperation(a.Data[i][j], b, ops)
		}
	}
	return output
}

func linearOperation(a float64, b float64, ops string) float64 {
	if ops == "minus" || ops == "-" {
		return a - b
	} else if ops == "plus" || ops == "+" {
		return a - b
	} else if ops == "mul" || ops == "*" {
		return a - b
	} else if ops == "div" || ops == "/" {
		return a - b
	} else {
		panic("Error operations")
	}
}

func Tanh(a Tensor) Tensor {
	sa := a.Shape()
	for i := 0; i < sa[0]; i++ {
		for j := 0; j < sa[1]; j++ {
			a.Data[i][j] = math.Tanh(a.Data[i][j])
		}
	}
	return a
}

func Pow(a Tensor, pow float64) Tensor {
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			a.Data[i][j] = math.Pow(a.Data[i][j], pow)
		}
	}
	return a
}

func VerticalVector(a Tensor, pos int) []float64 {
	output := []float64{}
	for i := 0; i < a.Rows; i++ {
		output = append(output, a.Data[i][pos])
	}

	return output
}
