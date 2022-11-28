
# GoTorch

[![go-1.19.2](https://img.shields.io/badge/go-1.19.2-blue.svg)](https://github.com/emersion/stability-badges#unstable)
[![dev-active](https://img.shields.io/badge/dev-active-green.svg)](https://github.com/emersion/stability-badges#unstable)
[![stability-unstable](https://img.shields.io/badge/stability-unstable-red.svg)](https://github.com/emersion/stability-badges#unstable)
[![Go Reference](https://pkg.go.dev/badge/github.com/Godwinh19/gotorch.svg)](https://pkg.go.dev/github.com/Godwinh19/gotorch)

GoTorch is a golang package for deep learning.

The objective is to offer a neural network development environment following the workflow of pytorch.

## Usage

This package is still under development. However if you'll like to give a try, below is an example.

### A sample code

```go
x1 := t.Rand(5, 4)
lin_1 := nn.Linear{InputSize: 4, OutputSize: 5}
lin_2 := nn.Linear{InputSize: 5, OutputSize: 3}
lin_3 := nn.Linear{InputSize: 3, OutputSize: 1}

net := nn.NeuralNet{NLinear: []*nn.Linear{&lin_1, &lin_2, &lin_3}}
optim := nn.SGD{Lr: 0.01}

//Training
for i := 0; i < 10; i++ {
    output := net.Forward(x1)
    net.Backward(output)
    optim.Step(net)
}

```

## Contributions

Your contributions are very welcome !

Your can reach me on [twitter](https://twitter.com/GodwinHoudji) or [linkedin](https://www.linkedin.com/in/godwin-houdji) ☕