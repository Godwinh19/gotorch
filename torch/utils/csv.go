package utils

import (
	"encoding/csv"
	"gotorch/torch/tensor"
	"log"
	"math/rand"
	"os"
	"strconv"
)

func ReadCsvFile(filePath string) tensor.Tensor {
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file "+filePath, err)
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal("Unable to parse file as CSV for "+filePath, err)
	}

	recordsTensor := convertToTensor(records)

	return recordsTensor
}

func convertToTensor(records [][]string) tensor.Tensor {
	rows := len(records)
	cols := len(records[0])

	out := tensor.Zeros(rows, cols)
	var err error
	for i, record := range records {
		for j, v := range record {
			out.Data[i][j], err = strconv.ParseFloat(v, 64)

			if err != nil {
				panic(err)
			}
		}
	}
	return out
}

func SplitXandY(records tensor.Tensor) (tensor.Tensor, tensor.Tensor, error) {
	//This function assume Y is last column and X the first ones

	shape := records.Shape()
	x := tensor.Zeros(shape[0], shape[1]-1)
	y := tensor.Zeros(shape[0], 1)

	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]-1; j++ {
			x.Data[i][j] = records.Data[i][j]
		}
		y.Data[i][0] = records.Data[i][shape[1]-1]
	}

	return x, y, nil
}

func Random(records tensor.Tensor, n int) tensor.Tensor {
	shape := records.Shape()
	newTensor := tensor.Zeros(n, shape[1])

	for i := 0; i < n; i++ {
		newTensor.Data[i] = records.Data[rand.Intn(shape[1])]
	}

	return newTensor
}
