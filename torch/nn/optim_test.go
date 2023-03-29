package nn

import (
	"fmt"
	"github.com/Godwinh19/gotorch/torch/tensor"
	"testing"
)

func TestSGD(t *testing.T) {
	data := tensor.Rand(4, 2)
	linear := Linear{InputSize: 2, OutputSize: 4}
	one_forward, _ := linear.Forward(data)
	/*
		@TODO
		compute manually forward vector for comparison
	*/
	fmt.Println("One forward", one_forward)

}
