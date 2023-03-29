package main

import (
	"fmt"
	t "github.com/Godwinh19/gotorch/torch/tensor"
)

func mainTensor() {
	ts := t.NewTensor([][]float64{{1., 2.0, 4}, {1.0, 5., 3.}})
	tr := t.Rand(1, 4)
	trb := t.Rand(4, 4)
	fmt.Println(ts.Shape())
	fmt.Println(tr)
	u := t.Dot(tr, trb)
	fmt.Println("Dot", u)
}
