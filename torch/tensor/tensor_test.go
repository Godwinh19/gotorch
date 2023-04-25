package tensor

import (
	"testing"
)

func TestTensor(t *testing.T) {
	// Test Shape
	data := []float64{1, 2, 3, 4, 5, 6}
	tensor := &Tensor{
		Dim:          []int{2, 3},
		Stride:       []int{3, 1},
		Data:         data,
		RequiredGrad: false,
	}

	shape := tensor.Shape()
	if shape[0] != 2 || shape[1] != 3 {
		t.Errorf("Expected Shape() to return [2 3], but got %v", shape)
	}

	// Test Size
	size := tensor.Size()
	if size != 6 {
		t.Errorf("Expected Size() to return 6, but got %d", size)
	}

	// Test Set and Get
	tensor.Set([]int{1, 1}, 10)
	val := tensor.Get([]int{1, 1})
	if val != 10 {
		t.Errorf("Expected Get([1, 1]) to return 10, but got %f", val)
	}

	// Test Index
	idx := tensor.Index([]int{1, 1})
	if idx != 4 {
		t.Errorf("Expected Index([1, 1]) to return 4, but got %d", idx)
	}

	// Test Zeros
	zeros := Zeros([]int{2, 2})
	for _, val := range zeros.Data {
		if val != 0 {
			t.Errorf("Expected Zeros() to return a tensor of all zeros, but found non-zero value: %f", val)
		}
	}

	// Test Ones
	ones := Ones([]int{2, 2})
	for _, val := range ones.Data {
		if val != 1 {
			t.Errorf("Expected Ones() to return a tensor of all ones, but found non-one value: %f", val)
		}
	}

	// Test Rand
	randTensor := Rand([]int{2, 2})
	for _, val := range randTensor.Data {
		if val < 0 || val >= 1 {
			t.Errorf("Expected Rand() to return a tensor with values between 0 and 1, but found value: %f", val)
		}
	}
}
