package tensor

import (
	"errors"
	"fmt"
	"reflect"
)

func IsSameShape(a, b []int) interface{} {
	if Prod(a) != Prod(b) {
		return errors.New("Matrix haven't same dimension")
	}
	return nil
}

func Prod(a []int) int {
	p := 1
	for _, x := range a {
		p *= x
	}
	return p
}

func IsEqualTo(a, b Tensor) bool {
	err := IsSameShape(a.Shape(), b.Shape())
	if err != nil {
		return false
	}

	if reflect.DeepEqual(a.Data, b.Data) {
		return true
	}
	return false
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
