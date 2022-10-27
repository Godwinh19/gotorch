package tensor

import "testing"

var ta = Zeros(2, 3)

//var tb = Zeros(4, 3)

func TestAddScalar(t *testing.T) {
	actual := AddScalar(ta, 2.0)
	expected := [][]float64{{2., 2., 2.}, {2., 2., 2.}}

	for i := 0; i < len(expected); i++ {
		for j := 0; j < len(expected[i]); j++ {
			if actual.Data[i][j] != expected[i][j] {
				t.Errorf("Expected (%+v) is not same as"+
					" actual (%+v)", expected, actual.Data)
			}
		}
	}
}
