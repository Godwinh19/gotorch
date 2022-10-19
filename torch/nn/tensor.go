package tensor

/*
A tensor is a n-dimensional array 
*/ 

type Tensor struct {
	data [][]float64
	requires_grad bool
	shape []float64
	depends_on Dependency
}

type Dependency struct {
	tensor Tensor
	//grad_fn function	
}

func (d *Dependency) grad_fn() Tensor {
	continue
}