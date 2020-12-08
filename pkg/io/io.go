package io

import (
	"encoding/csv"
	"encoding/gob"
	"fmt"
	"golem/pkg/model"
	"io"
	"os"
	"strconv"

	"github.com/nlpodyssey/spago/pkg/mat"
)

type DataBatch struct {
	// Features contain both the continuous features and the current learned
	// representation of the categorical features
	Features []mat.Matrix

	// CategoricalFeatures contain the indexes of the categorical features
	CategoricalFeatures [][]int

	// Targets contain the index or value of the target
	Targets []float64
}

func (d *DataBatch) Size() int {
	return len(d.Targets)
}

type Set map[string]struct{}

type DataParameters struct {
	TrainFile                string
	TargetColumn             string
	CategoricalColumns       Set
	BatchSize                int
	CategoricalEmbeddingSize int
}

// LoadData reads the train file and splits it into batches of at most BatchSize elements.
// TODO: add support for categorical features
func LoadData(p DataParameters) (*model.Metadata, []DataBatch, error) {

	inputFile, err := os.Open(p.TrainFile)
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

	metaData := model.NewMetadata()
	metaData.Columns = record
	metaData.CategoricalEmbeddingSize = p.CategoricalEmbeddingSize

	for i, col := range metaData.Columns {
		if col == p.TargetColumn {
			metaData.TargetColumn = i
			break
		}
	}
	if metaData.TargetColumn == -1 {
		return nil, nil, fmt.Errorf("target column %s not found in data header", p.TargetColumn)
	}

	featureIndex := 0

	for i, col := range metaData.Columns {
		_, isCategorical := p.CategoricalColumns[col]
		if i != metaData.TargetColumn {
			if !isCategorical {
				metaData.ContinuousFeaturesMap.Set(i, featureIndex)
				featureIndex++
			} else {
				metaData.CategoricalFeaturesMap.Set(i, featureIndex)
				featureIndex += p.CategoricalEmbeddingSize
			}
		}
	}

	var result []DataBatch
	currentBatch := DataBatch{}
	currentLine := 0
	featureSize := computeFeatureSize(metaData)

	for record, err = reader.Read(); err == nil; record, err = reader.Read() {
		//TODO: add support for continuous targets
		target := metaData.ParseCategoricalTarget(record[metaData.TargetColumn])
		if err != nil {
			return nil, nil, fmt.Errorf("error parsing target at line %d: %w", currentLine, err)
		}
		currentBatch.Targets = append(currentBatch.Targets, target)
		features := mat.NewEmptyVecDense(featureSize)
		//TODO: parse categorical features
		for column, index := range metaData.ContinuousFeaturesMap.ColumnToIndex {
			value, err := strconv.ParseFloat(record[column], 64)
			if err != nil {
				return nil, nil, fmt.Errorf("error parsing feature %s at line %d: %w", metaData.Columns[column], currentLine, err)
			}
			features.Set(index, 0, value)
		}

		currentBatch.Features = append(currentBatch.Features, features)

		if len(currentBatch.Targets) == p.BatchSize {
			result = append(result, currentBatch)
			currentBatch = DataBatch{}
		}
		currentLine++
	}

	if len(currentBatch.Targets) > 0 {
		result = append(result, currentBatch)
	}

	return metaData, result, nil
}

func computeFeatureSize(metaData *model.Metadata) int {
	return metaData.ContinuousFeaturesMap.Size() + metaData.CategoricalEmbeddingSize*metaData.CategoricalFeaturesMap.Size()

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
