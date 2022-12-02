package nn

import (
	"gotorch/torch/tensor"
	"math"
)

/*
Optimizer is used to adjust networks parameters of network based on gradients
computing during backpropagation
*/

type SGD struct {
	Lr float64
}

func (sgd *SGD) ZeroGradients(net NeuralNet) {
	for _, lin := range net.NLinear {
		lin.LLayer.Grads = nil
	}
}

func (sgd *SGD) Step(net NeuralNet) {
	for _, lin := range net.NLinear {
		sgd.updateParameters(lin.LLayer)
	}
}

func (sgd *SGD) updateParameters(params Layer) {
	params.Params["w"] = tensor.TensorOpsTensor(
		params.Params["w"],
		tensor.DotScalar(params.Grads["w"], sgd.Lr),
		"-")

	//params.Grads["b"] is a scalar then
	params.Params["b"] = tensor.TensorOpsScalar(
		params.Params["b"],
		tensor.DotScalar(params.Grads["b"], sgd.Lr).Data[0][0],
		"-")
}

// Learning rate scheduler
type LearningRateScheduler struct {
	// The starting learning rate
	startLearningRate float64
	// The number of steps in each epoch
	stepsPerEpoch int
	// The current step
	currentStep int
	// The learning rate decay factor
	decayFactor float64
}

// StepLRScheduler create a new learning rate scheduler with the given parameters
func StepLRScheduler(startLearningRate float64, stepsPerEpoch int, decayFactor float64) *LearningRateScheduler {
	return &LearningRateScheduler{
		startLearningRate: startLearningRate,
		stepsPerEpoch: stepsPerEpoch,
		currentStep: 0,
		decayFactor: decayFactor,
	}
}

// Next returns the next learning rate for the current step.
func (s *LearningRateScheduler) Next() float64 {
	// Calculate the learning rate for the current epoch
	currentEpoch := float64(s.currentStep) / float64(s.stepsPerEpoch)
	learningRate := s.startLearningRate * math.Pow(s.decayFactor, currentEpoch)
	// Increment the current step
	s.currentStep++
	return learningRate
}


