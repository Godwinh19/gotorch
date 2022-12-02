package nn

func ClipGrad(gradients []float64, maxValue float64) {
	for i, gradient := range gradients {
		if gradient > maxValue {
			gradients[i] = maxValue
		} else if gradient < - maxValue {
			gradients[i] = - maxValue
		}
	}
}