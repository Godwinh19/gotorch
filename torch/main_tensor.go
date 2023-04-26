package main

import (
	"fmt"
	t "github.com/Godwinh19/gotorch/torch/tensor"
)

func mainTensor() {
	// Create 2x3x5 tensor
	tensor := t.NewTensor([]int{2, 3, 5})
	tensor.Fill(1.)
	tensor.Print()
	new_t, err := tensor.Reshape([]int{2, 5, 3})
	if err != nil {
		fmt.Println(err)
	} else {
		new_t.Print()
	}
	tensor.Set([]int{1, 2, 4}, 2.0)
	fmt.Println(tensor.Get([]int{1, 2, 4}))
	tensor.Print()

}

func main() {
	mainTensor()
}
