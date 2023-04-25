package tensor

import (
	"testing"
)

func TestMul(t *testing.T) {
	a := Tensor{Dim: []int{2, 2}, Data: []float64{1, 2, 3, 4}}
	b := Tensor{Dim: []int{2, 2}, Data: []float64{5, 6, 7, 8}}
	expected := Tensor{Dim: []int{2, 2}, Data: []float64{5, 12, 21, 32}}
	result, err := Mul(a, b)
	if err != nil {
		t.Errorf("Mul() error = %v", err)
		return
	}
	if !IsTensorEqual(*result, expected) {
		t.Errorf("Mul() result = %v, expected = %v", result, expected)
	}
}

func TestDot(t *testing.T) {
	a := Tensor{Dim: []int{2, 3}, Data: []float64{1, 2, 3, 4, 5, 6}}
	b := Tensor{Dim: []int{3, 2}, Data: []float64{7, 8, 9, 10, 11, 12}}
	expected := Tensor{Dim: []int{2, 2}, Data: []float64{58, 64, 139, 154}}
	result, err := Dot(a, b)
	if err != nil {
		t.Errorf("Dot() error = %v", err)
		return
	}
	if !IsTensorEqual(*result, expected) {
		t.Errorf("Dot() result = %v, expected = %v", result, expected)
	}
}

func TestSum(t *testing.T) {
	a := Tensor{Dim: []int{2, 2}, Data: []float64{1, 2, 3, 4}}
	b := Tensor{Dim: []int{1, 2}, Data: []float64{5, 6}}
	expected := Tensor{Dim: []int{2, 2}, Data: []float64{6, 8, 8, 10}}
	result, err := Sum(a, b)
	if err != nil {
		t.Errorf("Sum() error = %v", err)
		return
	}
	if !IsTensorEqual(*result, expected) {
		t.Errorf("Sum() result = %v, expected = %v", result, expected)
	}
}

func TestSumWithBroadcasting(t *testing.T) {
	a := Tensor{Dim: []int{2, 2, 2}, Data: []float64{1, 2, 3, 4, 1, 2, 3, 4}}
	b := Tensor{Dim: []int{2}, Data: []float64{5, 6}}
	expected := Tensor{Dim: []int{2, 2, 2}, Data: []float64{6, 8, 8, 10, 6, 8, 8, 10}}
	result, err := Sum(a, b)
	if err != nil {
		t.Errorf("Sum() error = %v", err)
		return
	}
	if !IsTensorEqual(*result, expected) {
		t.Errorf("Sum() result = %v, expected = %v", result, expected)
	}
}
