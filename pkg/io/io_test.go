package io

import (
	"math"
	"math/rand"
	"testing"

	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/require"

	"golem/pkg/model"
)

func TestLoadData(t *testing.T) {
	params := DataParameters{
		DataFile:           "../../datasets/breast_cancer/breast-cancer.train",
		TargetColumn:       "Class",
		CategoricalColumns: NewSet("Class", "Age", "Menopause", "Tumor-size", "Inv-nodes", "Node-caps", "Breast", "Breast-quad", "Irradiat"),
		BatchSize:          10,
	}

	metaData, dataSet, dataErrors, err := LoadData(params, nil)
	require.NoError(t, err)
	require.Equal(t, 229, len(dataSet.Data))
	require.NotNil(t, metaData)
	require.Equal(t, 0, len(dataErrors))

	require.Equal(t, 1, dataSet.Data[0].ContinuousFeatures.Rows())
	require.Equal(t, 8, len(dataSet.Data[0].CategoricalFeatures))

	params.DataFile = "../../datasets/breast_cancer/breast-cancer.test"
	testMetaData, dataSet, dataErrors, err := LoadData(params, metaData)
	require.NoError(t, err)
	require.Equal(t, metaData, testMetaData)
	require.Equal(t, 1, len(dataErrors)) // Line 8 contains a category value for Age not present in training
	require.Equal(t, 56, len(dataSet.Data))
}

func TestDataSet(t *testing.T) {
	data := make([]*DataRecord, 100)
	for i := range data {
		data[i] = &DataRecord{
			Target: mat.Float(i),
		}
	}

	ds := NewDataSet(data, 10)
	ds.Rand = rand.New(rand.NewSource(42))
	ds.ResetOrder(RandomOrder)
	order1 := make([]mat.Float, 0, 100)

	for b := ds.Next(); len(b) > 0; b = ds.Next() {
		for _, d := range b {
			order1 = append(order1, d.Target)
		}
	}
	ds.ResetOrder(RandomOrder)
	order2 := make([]mat.Float, 0, 100)

	for b := ds.Next(); len(b) > 0; b = ds.Next() {
		for _, d := range b {
			order2 = append(order2, d.Target)
		}
	}

	require.Equal(t, 100, len(order1))
	require.Equal(t, 100, len(order2))
	require.NotEqual(t, order1, order2)

	ds.ResetOrder(OriginalOrder)
	order3 := make([]mat.Float, 0, 100)

	for b := ds.Next(); len(b) > 0; b = ds.Next() {
		for _, d := range b {
			order3 = append(order3, d.Target)
		}
	}

	require.Equal(t, 100, len(order3))
	for i, v := range order3 {
		require.Equal(t, mat.Float(i), v)
	}

}

func TestDataSet_RandomSplit(t *testing.T) {
	data := make([]*DataRecord, 1000)
	for i := range data {
		data[i] = &DataRecord{
			Target: mat.Float(i),
		}
	}
	ds := NewDataSet(data, 10)
	ds.Rand = rand.New(rand.NewSource(42))

	splits := ds.RandomSplit(500, 500)
	require.Equal(t, 2, len(splits))
	require.Equal(t, 500, splits[0].Size())
	require.Equal(t, 500, splits[1].Size())

	order1 := extractOrder(splits[0])
	order2 := extractOrder(splits[1])

	require.NotEqual(t, order1, order2)

}

func extractOrder(split *DataSet) []mat.Float {
	order := make([]mat.Float, 0)
	for b := split.Next(); len(b) > 0; b = split.Next() {
		for _, d := range b {
			order = append(order, d.Target)
		}
	}
	return order
}

func Test_Standardization(t *testing.T) {
	params := DataParameters{
		DataFile:           "../../datasets/iris/iris.train",
		TargetColumn:       "species",
		CategoricalColumns: NewSet("species"),
		BatchSize:          10,
	}
	metaData, dataSet, dataErrors, err := LoadData(params, nil)
	require.NoError(t, err)
	require.Equal(t, 0, len(dataErrors))
	require.NotZero(t, dataSet.Size())
	require.NotNil(t, metaData)

	require.InDelta(t, averageValue(dataSet, valueForColumn(t, metaData, "petal_length")), 0.0, 1e-6)
	require.InDelta(t, averageValue(dataSet, valueForColumn(t, metaData, "sepal_length")), 0.0, 1e-6)
	require.InDelta(t, averageValue(dataSet, valueForColumn(t, metaData, "petal_width")), 0.0, 1e-6)
	require.InDelta(t, averageValue(dataSet, valueForColumn(t, metaData, "sepal_width")), 0.0, 1e-6)

	require.InDelta(t, stdDev(dataSet, valueForColumn(t, metaData, "petal_length")), 1.0, 1e-6)
	require.InDelta(t, stdDev(dataSet, valueForColumn(t, metaData, "sepal_length")), 1.0, 1e-6)
	require.InDelta(t, stdDev(dataSet, valueForColumn(t, metaData, "petal_width")), 1.0, 1e-6)
	require.InDelta(t, stdDev(dataSet, valueForColumn(t, metaData, "petal_width")), 1.0, 1e-6)

	dataSet.ResetOrder(OriginalOrder)
	for batch := dataSet.Next(); len(batch) > 0; batch = dataSet.Next() {
		for _, d := range batch {
			switch d.Target {
			case 0.0, 1.0, 2.0:
				continue
			default:
				t.Fatalf("invalid value found for target: %f", d.Target)

			}

		}
	}
}
func Test_Standardization_Target(t *testing.T) {
	params := DataParameters{
		DataFile:           "../../datasets/cholesterol/cholesterol-train.csv",
		TargetColumn:       "chol",
		CategoricalColumns: NewSet("sex", "cp", "fbs", "restecg", "exang", "slope", "thal"),
		BatchSize:          10,
	}
	metaData, dataSet, _, err := LoadData(params, nil)
	_, dataSet, _, err = LoadData(params, metaData) // again, to make sure we can reload the dataset with previous metadata
	require.NoError(t, err)
	require.NotZero(t, dataSet.Size())
	require.NotNil(t, metaData)

	for i, col := range metaData.Columns {
		switch col.Type {
		case model.Continuous:
			if i == metaData.TargetColumn {
				v := func(d *DataRecord) float64 {
					return float64(d.Target)
				}
				require.InDelta(t, averageValue(dataSet, v), 0, 1e-1)
				require.InDelta(t, stdDev(dataSet, v), 1.0, 1e-1)

				continue
			}
			require.InDelta(t, averageValue(dataSet, valueForColumn(t, metaData, col.Name)), 0, 0.2, "Col: %s", col.Name)
			require.InDelta(t, stdDev(dataSet, valueForColumn(t, metaData, col.Name)), 1.0, 1e-1, "Col: %s", col.Name)

		}
	}

}

func stdDev(ds *DataSet, v valueFunc) float64 {
	avg := averageValue(ds, v)
	ds.ResetOrder(OriginalOrder)
	stdDev := 0.0

	for batch := ds.Next(); len(batch) > 0; batch = ds.Next() {
		for _, d := range batch {
			diff := v(d) - avg
			stdDev += math.Pow(diff, 2)
		}
	}
	stdDev = math.Sqrt(stdDev / (float64(ds.Size())))
	return stdDev
}

type valueFunc func(d *DataRecord) float64

func valueForColumn(t *testing.T, metaData *model.Metadata, name string) valueFunc {
	return func(d *DataRecord) float64 {
		col := -1
		for i, n := range metaData.Columns {
			if n.Name == name {
				col = i
				break
			}
		}
		require.NotEqual(t, col, -1, "Cannot find column: %s", name)

		index, ok := metaData.ContinuousFeaturesMap.ColumnToIndex[col]
		require.True(t, ok, "Column %d is not continuous", col)
		return float64(d.ContinuousFeatures.At(index, 0))

	}
}
func averageValue(ds *DataSet, v valueFunc) float64 {
	ds.ResetOrder(OriginalOrder)
	avg := 0.0
	for batch := ds.Next(); len(batch) > 0; batch = ds.Next() {
		for _, d := range batch {
			avg += v(d)
		}
	}
	avg = avg / (float64(ds.Size()))
	return avg
}
