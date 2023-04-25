package tensor

import (
	"testing"
)

func TestIsSameShape(t *testing.T) {
	// Test same shape
	a := []int{2, 3, 4}
	b := []int{2, 3, 4}
	if !IsSameShape(a, b) {
		t.Error("Expected same shape, but got different shape")
	}

	// Test different shape
	c := []int{2, 3, 4}
	d := []int{2, 2, 4}
	if IsSameShape(c, d) {
		t.Error("Expected different shape, but got same shape")
	}
}

func TestProd(t *testing.T) {
	// Test positive product
	a := []int{2, 3, 4}
	if Prod(a) != 24 {
		t.Error("Expected product of 24, but got something else")
	}

	// Test zero product
	b := []int{0, 3, 4}
	if Prod(b) != 0 {
		t.Error("Expected product of 0, but got something else")
	}
}

func TestIsTensorEqual(t *testing.T) {
	// Test equal tensors
	a := Tensor{Data: []float64{1, 2, 3}, Dim: []int{3}}
	b := Tensor{Data: []float64{1, 2, 3}, Dim: []int{3}}
	if !IsTensorEqual(a, b) {
		t.Error("Expected tensors to be equal, but got different tensors")
	}

	// Test tensors with different shape
	c := Tensor{Data: []float64{1, 2, 3}, Dim: []int{3}}
	d := Tensor{Data: []float64{1, 2, 3}, Dim: []int{2}}
	if IsTensorEqual(c, d) {
		t.Error("Expected tensors to be different, but got same tensors")
	}

	// Test tensors with different data
	e := Tensor{Data: []float64{1, 2, 3}, Dim: []int{3}}
	f := Tensor{Data: []float64{1, 2, 4}, Dim: []int{3}}
	if IsTensorEqual(e, f) {
		t.Error("Expected tensors to be different, but got same tensors")
	}
}

func TestIsIntArrayEqual(t *testing.T) {
	// Test equal slices
	a := []int{1, 2, 3}
	b := []int{1, 2, 3}
	if !IsIntArrayEqual(a, b) {
		t.Error("Expected slices to be equal, but got different slices")
	}

	// Test slices with different length
	c := []int{1, 2, 3}
	d := []int{1, 2}
	if IsIntArrayEqual(c, d) {
		t.Error("Expected slices to be different, but got same slices")
	}

	// Test slices with different values
	e := []int{1, 2, 3}
	f := []int{1, 2, 4}
	if IsIntArrayEqual(e, f) {
		t.Error("Expected slices to be different, but got same slices")
	}
}
