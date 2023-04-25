package tensor

import (
	"fmt"
	"reflect"
)

func IsSameShape(a, b []int) bool {
	if Prod(a) != Prod(b) {
		return false
	}
	return true
}

func Prod(a []int) int {
	p := 1
	for _, x := range a {
		p *= x
	}
	return p
}

func IsTensorEqual(a, b Tensor) bool {
	if !IsSameShape(a.Shape(), b.Shape()) {
		return false
	}

	if reflect.DeepEqual(a.Data, b.Data) {
		return true
	}
	return false
}

func IsIntArrayEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func Convert(val interface{}, typ reflect.Type) (interface{}, error) {
	// Get the value's current type
	valType := reflect.TypeOf(val)

	// If the value is already of the target type, return it unchanged
	if valType == typ {
		return val, nil
	}

	// Try to convert the value to the target type
	newVal := reflect.ValueOf(val).Convert(typ).Interface()

	// If the conversion fails, return an error
	if reflect.TypeOf(newVal) != typ {
		return nil, fmt.Errorf("cannot convert %s to %s", valType, typ)
	}

	return newVal, nil
}
