package tensor

import "math"

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
	la, lb := len(a), len(b)
	sum := 0.0
	for i := 0; i < la; i++ {
		for j := 0; j < lb; j++ {
			sum += a[i] * b[j]
		}
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
		return Tensor{Data: [][]float64{{sum}}, Rows: a.Rows, Cols: a.Cols}
	} else {
		a, b := params[0], params[1]
		output := Zeros(a.Rows, b.Cols)
		for i := 0; i < a.Rows; i++ {
			for j := 0; j < b.Cols; j++ {
				output.Data[i][j] = a.Data[i][j] + b.Data[0][j] //b is 1-d data
			}
		}
		return output
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
