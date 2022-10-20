package nn

/*
A tensor is a n-dimensional array 
*/ 

import (
	"gonum.org/v1/gonum/mat"
)

type Tensor struct {
	Data mat.Dense
	Requires_grad bool
	Shape []int64
	//depends_on Dependency
}

type Dependency struct {
	Tensor Tensor
	//grad_fn function	
}

func (d *Dependency) grad_fn() {
	//
}