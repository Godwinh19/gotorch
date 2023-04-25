package nn

import (
	"fmt"
	"github.com/Godwinh19/gotorch/torch/tensor"
	"testing"
)

func TestForward(t *testing.T) {
	data := *tensor.Rand([]int{5, 4})
	linear_1 := Linear{InputSize: 4, OutputSize: 5, Inputs: data}
	linear_2 := Linear{InputSize: 5, OutputSize: 3, Inputs: data}

	one_forward, _ := linear_1.Forward(data)
	out, _ := linear_2.Forward(one_forward)

	fmt.Println(out.Shape())
	fmt.Println("Value after 2 forward", out)
}
