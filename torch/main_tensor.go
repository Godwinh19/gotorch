package main

import (
	"fmt"
	t "github.com/Godwinh19/gotorch/torch/tensor"
)

func mainTensor() {
	ts := t.NewTensor([]int{3, 2})
	tr := t.Rand([]int{1, 4})
	trb := t.Rand([]int{4, 4})
	fmt.Println(ts.Shape())
	fmt.Println(tr)
	u, _ := t.Dot(*tr, *trb)
	fmt.Println("Dot", u)
	u, _ = t.Dot(*t.Ones([]int{3, 3}), *t.Ones([]int{3, 2}))
	fmt.Println("Dot ones", u)
}

func main() {
	mainTensor()
}
