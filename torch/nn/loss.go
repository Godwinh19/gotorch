package nn

/*
A loss function measure how good predictions are, it's used to adjust
the parameters of our network
*/
import (
	"gonum.org/v1/gonum/mat"
	"math"
)

type Loss struct {
	Predicted Tensor
	Actual Tensor
}

func (l *Loss) MSELoss() float64 {
	//numgo module for sum
	return mat.Sum(math.Pow((l.Predicted.data - l.Actual.data), 2))
}

func (l *Loss) MSEGrad() float64 {
	return 2 * (l.Predicted.data - l.Actual.data)
}