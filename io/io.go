package io

import (
	"encoding/csv"
	"fmt"
	"os"

	"github.com/nlpodyssey/spago/pkg/mat"
)

type DatasetMetadata struct {
	Columns []string
}
type dataBatch struct {
	continuousFeatures  []mat.Dense
	categoricalFeatures [][]int
}

// loadData reads the train file and splits it into batches of at most batchSize elements.
// TODO: add support for categorical features
func loadData(trainFile string, batchSize int) (*datasetMetadata, []dataBatch, error) {

	inputFile, err := os.Open(trainFile)
	if err != nil {
		return nil, nil, fmt.Errorf("error opening training file: %w", err)
	}

	reader := csv.NewReader(inputFile)
	reader.Comma = ','

	//First line is expected to be a header
	record, err := reader.Read()
	if err != nil {
		return nil, nil, fmt.Errorf("error reading data header: %w", err)
	}

	metaData := DatasetMetadata{record}

	for record, err = reader.Read(); err != nil; record, err = reader.Read() {

	}

	return &metaData, nil, nil

}
func train(trainFile, outputFile string) {

}
