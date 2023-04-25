package tensor

import (
	"testing"
)

func TestTranspose(t *testing.T) {
	// Create a 2x3x4 tensor
	t1 := NewTensor([]int{2, 3, 4})

	// Initialize the data with some arbitrary values
	for i := 0; i < len(t1.Data); i++ {
		t1.Data[i] = float64(i)
	}

	// Transpose the tensor
	t2 := t1.Transpose()

	// Make sure the dimensions are now 4x3x2
	expectedDim := []int{4, 3, 2}
	if !IsIntArrayEqual(t2.Dim, expectedDim) {
		t.Errorf("Expected tensor with dimensions %v, but got dimensions %v", expectedDim, t2.Dim)
	}

	// Make sure the values are in the correct order
	// The first element of the original tensor should be the first element of the new tensor,
	// which is at index (0,0,0) in the new tensor
	expectedValue := float64(0)
	if t2.Data[0] != expectedValue {
		t.Errorf("Expected value %v at index (0,0,0), but got value %v", expectedValue, t2.Data[0])
	}

	// The last element of the original tensor should be the last element of the new tensor,
	// which is at index (3,2,1) in the new tensor
	expectedValue = float64(len(t1.Data) - 1)
	if t2.Data[len(t2.Data)-1] != expectedValue {
		t.Errorf("Expected value %v at index (3,2,1), but got value %v", expectedValue, t2.Data[len(t2.Data)-1])
	}
}
