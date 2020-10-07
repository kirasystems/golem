package io

import (
	"encoding/csv"
	"encoding/gob"
	"fmt"
	"golem/model"
	"io"
	"os"
	"strconv"

	"github.com/nlpodyssey/spago/pkg/mat"
)

type DataBatch struct {
	ContinuousFeatures  []mat.Matrix
	CategoricalFeatures [][]int
	Targets             []float64
}

func (d *DataBatch) Size() int {
	return len(d.Targets)
}

// LoadData reads the train file and splits it into batches of at most batchSize elements.
// TODO: add support for categorical features
func LoadData(trainFile, targetColumn string, batchSize int) (*model.Metadata, []DataBatch, error) {

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

	metaData := model.Metadata{
		Columns:                record,
		ContinuousFeaturesMap:  map[int]int{},
		CategoricalFeaturesMap: map[int]int{},
		TargetColumn:           -1,
	}

	for i, col := range metaData.Columns {
		if col == targetColumn {
			metaData.TargetColumn = i
			break
		}
	}
	if metaData.TargetColumn == -1 {
		return nil, nil, fmt.Errorf("target column %s not found in data header", targetColumn)
	}

	ind := 0
	//TODO: add support for categorical features
	for i := range metaData.Columns {
		if i != metaData.TargetColumn {
			metaData.ContinuousFeaturesMap[i] = ind
			ind++
		}
	}

	var result []DataBatch
	currentBatch := DataBatch{}
	currentLine := 0

	for record, err = reader.Read(); err == nil; record, err = reader.Read() {
		target := metaData.ParseCategoricalTarget(record[metaData.TargetColumn])
		if err != nil {
			return nil, nil, fmt.Errorf("error parsing target at line %d: %w", currentLine, err)
		}
		currentBatch.Targets = append(currentBatch.Targets, target)
		continuousFeatures := mat.NewEmptyVecDense(len(metaData.ContinuousFeaturesMap))
		//TODO: parse categorical features
		for c1, c2 := range metaData.ContinuousFeaturesMap {
			value, err := strconv.ParseFloat(record[c1], 64)
			if err != nil {
				return nil, nil, fmt.Errorf("error parsing feature %s at line %d: %w", metaData.Columns[c1], currentLine, err)
			}
			continuousFeatures.Set(c2, 0, value)
		}

		currentBatch.ContinuousFeatures = append(currentBatch.ContinuousFeatures, continuousFeatures)

		if len(currentBatch.Targets) == batchSize {
			result = append(result, currentBatch)
			currentBatch = DataBatch{}
		}
		currentLine++
	}

	if len(currentBatch.Targets) > 0 {
		result = append(result, currentBatch)
	}

	return &metaData, result, nil
}

func SaveModel(model *model.Model, writer io.Writer) error {
	encoder := gob.NewEncoder(writer)
	err := encoder.Encode(model)
	if err != nil {
		return fmt.Errorf("error encoding model: %w", err)
	}
	return nil
}

func LoadModel(input io.Reader) (*model.Model, error) {
	decoder := gob.NewDecoder(input)
	model := model.Model{}
	err := decoder.Decode(&model)
	if err != nil {
		return nil, fmt.Errorf("error decoding model: %w", err)
	}
	return &model, nil

}
