package main

import "fmt"
import t "gotorch/torch/tensor"

func main() {
	ts := t.NewTensor([][]float64{{1.,2.0, 4}, {1.0, 5., 3.}})
	tr := t.Rand(5,4)
	fmt.Println(ts.Shape())
	fmt.Println(tr)
}