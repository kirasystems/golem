package io

import (
	"math/rand"
	"testing"

	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/require"
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
