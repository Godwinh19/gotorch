package nn

import(
	"testing"
	"fmt"
	"gotorch/torch/tensor"
)

func TestSGD(t *testing.T){
	data := tensor.Rand(4,2)
	linear := Linear{InputSize: 2, OutputSize: 4}
	one_forward := linear.Forward(data)
	/*
	@TODO
	compute manually forward vector for comparison
	*/
	fmt.Println("One forward", one_forward)

}