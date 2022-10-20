package tensor


func AssertTensorFormat(t Tensor) error {
	rows := cap(t.Data)
	cols := len(t.Data[0])
	for i := 0; i<rows; i++ {
		if len(t.Data[i]) != cols{
			panic("tensor: mismatched dimension")
		}
	}
	return nil
}