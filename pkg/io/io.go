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

type void struct{}

var Void = void{}

type Set map[string]void

func NewSet(values ...string) Set {
	set := Set{}
	for _, val := range values {
		set[val] = Void
	}
	return set
}

type DataParameters struct {
	DataFile                 string
	TargetColumn             string
	CategoricalColumns       Set
	BatchSize                int
	CategoricalEmbeddingSize int
}

type DataError struct {
	Line  int
	Error string
}

// LoadData reads the train file and splits it into batches of at most BatchSize elements.
func LoadData(p DataParameters, metaData *model.Metadata) (*model.Metadata, []DataBatch, []DataError, error) {

	var errors []DataError
	inputFile, err := os.Open(p.DataFile)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("error opening file: %w", err)
	}

	reader := csv.NewReader(inputFile)
	reader.Comma = ','

	//First line is expected to be a header
	record, err := reader.Read()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("error reading data header: %w", err)
	}

	newMetadata := false
	if metaData == nil {
		metaData = model.NewMetadata()
		newMetadata = true
		metaData.Columns = record
		metaData.CategoricalEmbeddingSize = p.CategoricalEmbeddingSize
		if err := setTargetColumn(p, metaData); err != nil {
			return nil, nil, nil, err
		}
		buildFeatureIndex(p, metaData)
	}

	var result []DataBatch
	currentBatch := DataBatch{}
	currentLine := 0
	featureSize := computeFeatureSize(metaData)

	for record, err = reader.Read(); err == nil; record, err = reader.Read() {
		//TODO: add support for continuous targets
		targetValue, err := parseTarget(newMetadata, metaData, record[metaData.TargetColumn])
		if err != nil {
			errors = append(errors, DataError{
				Line:  currentLine,
				Error: err.Error(),
			})
			continue
		}

		currentBatch.Targets = append(currentBatch.Targets, targetValue)
		features := mat.NewEmptyVecDense(featureSize)

		err = parseContinuousFeatures(metaData, record, features)
		if err != nil {
			errors = append(errors, DataError{
				Line:  currentLine,
				Error: err.Error(),
			})
			continue
		}

		categoricalFeatures, err := parseCategoricalFeatures(metaData, newMetadata, record)
		if err != nil {
			errors = append(errors, DataError{
				Line:  currentLine,
				Error: err.Error(),
			})
			continue
		}

		currentBatch.Features = append(currentBatch.Features, features)
		currentBatch.CategoricalFeatures = append(currentBatch.CategoricalFeatures, categoricalFeatures)

		if len(currentBatch.Targets) == p.BatchSize {
			result = append(result, currentBatch)
			currentBatch = DataBatch{}
		}
		currentLine++
	}

	if len(currentBatch.Targets) > 0 {
		result = append(result, currentBatch)
	}

	return metaData, result, errors, nil
}

func parseCategoricalFeatures(metaData *model.Metadata, newMetadata bool, record []string) ([]int, error) {
	categoricalFeatures := make([]int, 0, metaData.CategoricalFeaturesMap.Size())
	for column, index := range metaData.CategoricalFeaturesMap.ColumnToIndex {
		categoryNameMap, ok := metaData.CategoricalFeaturesValuesMap[index]
		if !ok {
			if newMetadata {
				categoryNameMap = model.NewNameMap()
				metaData.CategoricalFeaturesValuesMap[index] = categoryNameMap
			} else {
				return nil, fmt.Errorf("unknown categorical attribute %s (should not happen!)", metaData.Columns[column])
			}

		}
		categoryValue := 0
		if newMetadata {
			categoryValue = categoryNameMap.ValueFor(record[column])
		} else {
			categoryValue, ok = categoryNameMap.NameToIndex[record[column]]
			if !ok {
				return nil, fmt.Errorf("unknown value %s for categorical attribute %s", metaData.Columns[column], record[column])
			}
		}
		categoricalFeatures = append(categoricalFeatures, categoryValue)
	}
	return categoricalFeatures, nil
}

func parseContinuousFeatures(metaData *model.Metadata, record []string, features *mat.Dense) error {
	for column, index := range metaData.ContinuousFeaturesMap.ColumnToIndex {
		value, err := strconv.ParseFloat(record[column], 64)
		if err != nil {
			return fmt.Errorf("error parsing feature %s: %w", metaData.Columns[column], err)
		}
		features.Set(index, 0, value)
	}
	return nil
}

func parseTarget(newMetadata bool, metaData *model.Metadata, target string) (float64, error) {
	targetValue := 0.0
	if newMetadata {
		targetValue = metaData.ParseOrAddCategoricalTarget(target)
	} else {
		var ok bool
		targetValue, ok = metaData.ParseCategoricalTarget(target)
		if !ok {
			return 0, fmt.Errorf("unknown categorical targetValue value %s", target)
		}
	}

	return targetValue, nil
}

func buildFeatureIndex(p DataParameters, metaData *model.Metadata) {
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
}

func setTargetColumn(p DataParameters, metaData *model.Metadata) error {
	for i, col := range metaData.Columns {
		if col == p.TargetColumn {
			metaData.TargetColumn = i
			return nil
		}
	}
	return fmt.Errorf("target column %s not found in data header", p.TargetColumn)
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
