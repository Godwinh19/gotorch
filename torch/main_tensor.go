package main

import (
	"fmt"
	t "github.com/Godwinh19/gotorch/torch/tensor"
)

func mainTensor() {
	// Create 2x3x5 tensor
	tensor := t.NewTensor([]int{2, 3, 5})
	tensor.Fill(2.) // fill the tensor with 2.0
	tensor.Print()  // Print tensor

	newT, err := tensor.Reshape([]int{2, 5, 3}) // Reshape tensor to 2x5x3
	if err == nil {
		newT.Print()
	}
	tensor.Set([]int{1, 2, 4}, 2.0)
	fmt.Println(tensor.Get([]int{1, 2, 4}))
	tensor.Print()

	// Broadcasting
	a := t.Ones([]int{2, 4})
	b := t.Ones([]int{4})
	c, err := t.Sum(*a, *b)
	if err == nil {
		c.Print()
	}

}

func main() {
	mainTensor()
}
