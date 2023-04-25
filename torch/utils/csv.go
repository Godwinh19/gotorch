package utils

import (
	"encoding/csv"
	"errors"
	"github.com/Godwinh19/gotorch/torch/tensor"
	"log"
	"math/rand"
	"os"
	"strconv"
)

func ReadCsvFile(filePath string) tensor.Tensor {
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file " + filePath + ": " + err.Error())
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal("Unable to parse file as CSV for " + filePath + ": " + err.Error())
	}

	recordsTensor := convertToTensor(records)

	return recordsTensor
}

func convertToTensor(records [][]string) tensor.Tensor {
	rows := len(records)
	cols := len(records[0])

	out := tensor.NewTensor([]int{rows, cols})
	for i, record := range records {
		for _, v := range record {
			val, err := strconv.ParseFloat(v, 64)

			if err != nil {
				panic(err)
			}
			out.Data[i] = val
		}
	}
	return *out
}

func SplitXandY(records tensor.Tensor) (tensor.Tensor, tensor.Tensor, error) {
	//This function assume Y is last column and X the first ones

	shape := records.Shape()
	if shape[1] < 2 {
		return tensor.Tensor{}, tensor.Tensor{}, errors.New("expected at least 2 columns in input tensor")
	}
	x := tensor.Zeros([]int{shape[0], shape[1] - 1})
	y := tensor.Zeros([]int{shape[0], 1})
	idx, idy := 0, 0

	for i := 0; i < shape[0]; i++ {
		if i%shape[1] == 0 {
			y.Data[idy] = records.Data[i]
			idy++
		} else {
			x.Data[idx] = records.Data[i]
			idx++
		}
	}

	return *x, *y, nil
}

func Random(records tensor.Tensor, n int) tensor.Tensor {
	shape := records.Shape()
	if n > shape[0] {
		n = shape[0]
	}

	shuffled := make([]float64, shape[0])
	perm := rand.Perm(shape[0])
	for i, p := range perm {
		shuffled[i] = records.Data[p]
	}

	// Select first n rows
	out := tensor.Zeros([]int{n, shape[1]})
	for i := 0; i < n; i++ {
		out.Data[i] = shuffled[i]
	}

	return *out
}

func isArrayEqual(a, b []float64) bool {
	if len(a) == len(b) {
		for i := 0; i < len(a); i++ {
			if a[i] != b[i] {
				return false
			}
		}
		return true
	} else {
		return false
	}
}

func isNdArrayContainsArray(a [][]float64, b []float64) bool {
	for _, arr := range a {
		if isArrayEqual(arr, b) {
			return true
		}
	}
	return false
}
