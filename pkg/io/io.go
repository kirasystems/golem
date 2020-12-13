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

// DataInstance holds data for a single data point.
type DataRecord struct {

	// ContinuousFeatures contains the raw value of the continuous features
	// these are indexed according to the mapping from continuous feature to index
	// specified in the dataset metadata
	ContinuousFeatures mat.Matrix

	// CategoricalFeatures contain the category values for the categorical features.
	// Each column of the slice corresponds to a categorical feature. The mapping between
	// column indices and feature is specified in the dataset metadata
	CategoricalFeatures []int

	// Target contains the target value.
	// Float64 is used to represent valus for both continuous and categorical target types.

	Target float64
}

// DataBatch holds a minibatch of data.
type DataBatch []DataRecord

func (d DataBatch) Size() int {
	return len(d)
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
	DataFile           string
	TargetColumn       string
	CategoricalColumns Set
	BatchSize          int
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
		if err := setTargetColumn(p, metaData); err != nil {
			return nil, nil, nil, err
		}
		buildFeatureIndex(p, metaData)
	}

	var result []DataBatch
	currentBatch := make(DataBatch, 0, p.BatchSize)
	currentLine := 0

	for record, err = reader.Read(); err == nil; record, err = reader.Read() {
		//TODO: add support for continuous targets

		dataRecord := DataRecord{}

		targetValue, err := parseTarget(newMetadata, metaData, record[metaData.TargetColumn])
		if err != nil {
			errors = append(errors, DataError{
				Line:  currentLine,
				Error: err.Error(),
			})
			continue
		}

		dataRecord.Target = targetValue
		dataRecord.ContinuousFeatures = mat.NewEmptyVecDense(metaData.ContinuousFeaturesMap.Size())

		err = parseContinuousFeatures(metaData, record, dataRecord.ContinuousFeatures)
		if err != nil {
			errors = append(errors, DataError{
				Line:  currentLine,
				Error: err.Error(),
			})
			continue
		}

		dataRecord.CategoricalFeatures, err = parseCategoricalFeatures(metaData, newMetadata, record)
		if err != nil {
			errors = append(errors, DataError{
				Line:  currentLine,
				Error: err.Error(),
			})
			continue
		}
		currentBatch = append(currentBatch, dataRecord)
		if len(currentBatch) == p.BatchSize {
			result = append(result, currentBatch)
			currentBatch = make(DataBatch, 0, p.BatchSize)
		}
		currentLine++
	}

	if len(currentBatch) > 0 {
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

func parseContinuousFeatures(metaData *model.Metadata, record []string, features mat.Matrix) error {
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
	continuousFeatureIndex := 0
	categoricalFeatureIndex := 0
	for i, col := range metaData.Columns {
		_, isCategorical := p.CategoricalColumns[col]
		if i != metaData.TargetColumn {
			if !isCategorical {
				metaData.ContinuousFeaturesMap.Set(i, continuousFeatureIndex)
				continuousFeatureIndex++
			} else {
				metaData.CategoricalFeaturesMap.Set(i, categoricalFeatureIndex)
				categoricalFeatureIndex++
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
