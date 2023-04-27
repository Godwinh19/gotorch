# Package `tensor`

Package `tensor` provides a Tensor implementation in Go.

## `type Tensor struct`

`Tensor` represents a tensor with a given shape, stride, data, and gradient requirement.
- `Dim` field is the shape of the tensor
- `Stride` field is the stride of the tensor
- `Data` field is the data of the tensor, and
- `RequiredGrad` field is a boolean indicating if the gradient is required.

### `func (t *Tensor) Shape() []int`

`Shape` returns the shape of the tensor.

### `func (t *Tensor) Size() int`

`Size` returns the total number of elements in the tensor.

### `func NewTensor(dim []int) *Tensor`

`NewTensor` creates a new tensor with the given shape.

```go
// Create a new tensor with shape [3, 2]
t := tensor.NewTensor([]int{3, 2})

//Getting the shape of t
shape := t.Shape() // []int{3,2}

//Size of t
size := t.Size() // 6
```

### `func (t *Tensor) Set(index []int, val float64)`

`Set` sets the value of the tensor at the given index.

### `func (t *Tensor) Get(index []int) float64`

`Get` returns the value of the tensor at the given index.

```go
// Set the value at index [1, 1] to 5.0
t.Set([]int{1, 1}, 5.0)

// Get the value at index [1, 1]
val := t.Get([]int{1, 1}) // 5.0

```

### `func (t *Tensor) Index(index []int) int`

`Index` returns the index of the tensor at the given index.

```go
// Get the index of element at [1,1]
index := t.Index([]int{1, 1}) //1
```

### `func (t *Tensor) Add(other interface{}) *Tensor`

`Add` adds the given tensor or scalar to the tensor and returns a new tensor.

### `func (t *Tensor) Sub(other interface{}) *Tensor`

`Sub` subtracts the given tensor or scalar from the tensor and returns a new tensor.

### `func (t *Tensor) Mul(other interface{}) *Tensor`

`Mul` multiplies the given tensor or scalar to the tensor and returns a new tensor.

### `func (t *Tensor) Div(other interface{}) *Tensor`

`Div` divides the given tensor or scalar from the tensor and returns a new tensor.

### `func (t *Tensor) Fill(value float64)`

`Fill` fills the tensor with the given value.

```go
// Create a new tensor with shape [3, 2]
t2 := tensor.NewTensor([]int{3, 2})

// Fill t2 with value 2.0
t2.Fill(2.0)

// Add t2 to t and return a new tensor
t3 := t.Add(t2)

// Subtract t2 from t and return a new tensor
t4 := t.Sub(t2)

// Multiply t by 2.0 and return a new tensor
t5 := t.Mul(2.0)

// Divide t by 2.0 and return a new tensor
t6 := t.Div(2.0)

```

### `func Zeros(shape []int) *Tensor`

`Zeros` creates a new tensor of zeros with the given shape.

### `func Ones(shape []int) *Tensor`

`Ones` creates a new tensor of ones with the given shape.

### `func Rand(shape []int) *Tensor`

`Rand` creates a new tensor with random values between 0 and 1 with the given shape.

```go
// Create a new tensor of zeros with shape [2, 2]
t7 := tensor.Zeros([]int{2, 2})

// Create a new tensor of ones with shape [2, 2]
t8 := tensor.Ones([]int{2, 2})

// Create a new tensor with random values between 0 and 1 with shape [2, 2]
t9 := tensor.Rand([]int{2, 2})
```



### `func (t *Tensor) Softmax(dim int) (*Tensor, error)`

`Softmax` returns the softmax of the tensor along the given dimension.


## `func (t *Tensor) Transpose() *Tensor`

`Transpose` transposes the tensor and returns a new tensor.

```go
// Transpose tensor t and return a new tensor
t10 := t.Transpose()
```

### `func (t *Tensor) TensorAlmostEquals(b *Tensor) bool`

`TensorAlmostEquals` returns true if the difference between a and b is less than or equal to the given tolerance.

```go
// Create a new tensor with shape [3, 2]
t12 := tensor.NewTensor([]int{3, 2})

// Fill t12 with value 5.0
t12.Fill(5.0)

// Check if t and t12 are almost equal with tolerance 1e-6
is_equal := t.TensorAlmostEquals(t12)

```
