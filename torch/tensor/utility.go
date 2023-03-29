package tensor

import (
	"errors"
	"fmt"
	"reflect"
)

func AssertTensorFormat(t Tensor) error {
	rows := cap(t.Data)
	cols := len(t.Data[0])
	for i := 0; i < rows; i++ {
		if len(t.Data[i]) != cols {
			panic("tensor: mismatched dimension")
		}
	}
	return nil
}

func (t Tensor) SameTensorShape(tb Tensor) error {
	st := t.Shape()
	stb := tb.Shape()

	if (st[0] != stb[0]) || (st[1] != stb[1]) {
		return errors.New("Matrix haven't same dimension")
	}
	return nil
}

func (t Tensor) SameTensorRowColShape(tb Tensor) error {
	st := t.Shape()
	stb := tb.Shape()

	if st[1] != stb[0] {
		msg := fmt.Sprintf("Row and Column mismatched %v and %v", t.Shape(), tb.Shape())
		return errors.New(msg)
	}
	return nil
}

func (t Tensor) IsEqualTo(tb Tensor) bool {
	err := t.SameTensorShape(tb)
	if err != nil {
		return false
	}
	shape := t.Shape()
	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			if t.Data[i][j] != tb.Data[i][j] {
				return false
			}
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
