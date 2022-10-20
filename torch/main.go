package main

import (
	"gotorch/torch/nn"
	"fmt"
)

func main() {
	t := nn.Tensor{}
	l := nn.Loss{}
	fmt.Println("hello", t, l)
}