package main

import (
	"fmt"
	t "gotorch/torch/tensor"
)

func main() {
	ts := t.NewTensor([][]float64{{1., 2.0, 4}, {1.0, 5., 3.}})
	tr := t.Rand(1, 4)
	trb := t.Rand(4, 4)
	fmt.Println(ts.Shape())
	fmt.Println(tr)
	u := t.Dot(tr, trb)
	fmt.Println("Dot", u)
}
