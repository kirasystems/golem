package io

import (
	"encoding/csv"
	"encoding/gob"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"

	"golem/pkg/model"

	mat "github.com/nlpodyssey/spago/pkg/mat32"
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

	Target mat.Float
}

// DataBatch holds a minibatch of data.
type DataBatch []*DataRecord

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

type DataSet struct {
	Data         []*DataRecord
	BatchSize    int
	Rand         *rand.Rand
	currentOrder []int
	currentIndex int
}

type DatasetOrder int

const (
	OriginalOrder DatasetOrder = iota
	RandomOrder
)

func (d *DataSet) ResetOrder(order DatasetOrder) {
	switch order {
	case OriginalOrder:
		d.currentOrder = make([]int, len(d.Data))
		for i := range d.currentOrder {
			d.currentOrder[i] = i
		}
	case RandomOrder:
		d.currentOrder = d.Rand.Perm(len(d.Data))
	default:
		panic("invalid dataset order received " + strconv.Itoa(int(order)))
	}

	d.currentIndex = 0
}
func (d *DataSet) Next() DataBatch {
	batch := make(DataBatch, 0, d.BatchSize)
	for ; d.currentIndex < len(d.Data) && len(batch) < d.BatchSize; d.currentIndex++ {
		batch = append(batch, d.Data[d.currentOrder[d.currentIndex]])
	}
	return batch
}

func NewDataSet(data []*DataRecord, batchSize int) *DataSet {
	ds := &DataSet{Data: data, BatchSize: batchSize}
	return ds
}

// LoadData reads the train file and splits it into batches of at most BatchSize elements.
func LoadData(p DataParameters, metaData *model.Metadata) (*model.Metadata, *DataSet, []DataError, error) {

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
		metaData.Columns = parseColumns(record, p)
		if err := setTargetColumn(p, metaData); err != nil {
			return nil, nil, nil, err
		}
		buildFeatureIndex(metaData)
	}

	var data []*DataRecord
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
		data = append(data, &dataRecord)
		currentLine++
	}

	dataSet := NewDataSet(data, p.BatchSize)

	return metaData, dataSet, errors, nil
}

func parseColumns(record []string, p DataParameters) []model.Column {
	result := make([]model.Column, len(record))

	columnType := func(c string) model.ColumnType {
		if _, ok := p.CategoricalColumns[c]; ok {
			return model.Categorical
		}
		return model.Continuous
	}
	for i := range result {
		result[i] = model.Column{
			Name: record[i],
			Type: columnType(record[i]),
		}
	}

	return result
}

func parseCategoricalFeatures(metaData *model.Metadata, newMetadata bool, record []string) ([]int, error) {
	categoricalFeatures := make([]int, metaData.CategoricalFeaturesMap.Size())
	for column := range metaData.CategoricalFeaturesMap.Columns() {
		index := metaData.CategoricalFeaturesMap.ColumnToIndex[column]
		categoryValue := model.CategoricalValue{
			Column: column,
			Value:  record[column],
		}
		valueIndex := 0
		if newMetadata {
			valueIndex = metaData.CategoricalValuesMap.ValueFor(categoryValue)
		} else {
			ok := false
			valueIndex, ok = metaData.CategoricalValuesMap.ValueToIndex[categoryValue]
			if !ok {
				return nil, fmt.Errorf("unknown value %s for categorical attribute %s", record[column], metaData.Columns[column].Name)
			}
		}
		categoricalFeatures[index] = valueIndex
	}
	return categoricalFeatures, nil
}

func parseContinuousFeatures(metaData *model.Metadata, record []string, features mat.Matrix) error {
	for column, index := range metaData.ContinuousFeaturesMap.ColumnToIndex {
		value, err := strconv.ParseFloat(record[column], 64)
		if err != nil {
			return fmt.Errorf("error parsing feature %s: %w", metaData.Columns[column].Name, err)
		}
		features.Set(index, 0, mat.Float(value))
	}
	return nil
}

func parseTarget(newMetadata bool, metaData *model.Metadata, target string) (mat.Float, error) {

	var parseFunc func(string) (mat.Float, error)
	switch metaData.TargetType() {
	case model.Categorical:
		if newMetadata {
			parseFunc = metaData.ParseOrAddCategoricalTarget
		} else {
			parseFunc = metaData.ParseCategoricalTarget
		}
	case model.Continuous:
		parseFunc = metaData.ParseContinuousTarget
	}

	targetValue, err := parseFunc(target)
	if err != nil {
		return 0, fmt.Errorf("unable to parse target value %s: %w", target, err)
	}

	return targetValue, nil
}

func buildFeatureIndex(metaData *model.Metadata) {
	continuousFeatureIndex := 0
	categoricalFeatureIndex := 0
	for i, col := range metaData.Columns {
		if i != metaData.TargetColumn {
			if col.Type == model.Continuous {
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
		if col.Name == p.TargetColumn {
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
	m := model.Model{}
	err := decoder.Decode(&m)
	if err != nil {
		return nil, fmt.Errorf("error decoding model: %w", err)
	}
	return &m, nil
}
