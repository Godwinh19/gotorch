package utils

import (
	"path/filepath"
	"testing"
)

var data, _ = filepath.Abs("../../examples/data/iris.csv")
var records = ReadCsvFile(data)

func TestCSVLoad(t *testing.T) {
	x, _, _ := SplitXandY(records)
	shape := x.Shape()
	if shape[0] != 150 || shape[1] != 4 {
		t.Errorf("Expected [150, 4] but got %v", x.Shape())
	}
}

func TestRandom(t *testing.T) {
	n := 5
	random := Random(records, n)

	if random.Shape()[0] != n {
		t.Errorf("Expected %v but got %v", n, random.Shape())
	}
}
