package io

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLoadData(t *testing.T) {
	params := DataParameters{
		DataFile:           "../../datasets/breast_cancer/breast-cancer.train",
		TargetColumn:       "Class",
		CategoricalColumns: NewSet("Age", "Menopause", "Tumor-size", "Inv-nodes", "Node-caps", "Breast", "Breast-quad", "Irradiat"),
		BatchSize:          10,
	}

	metaData, data, dataErrors, err := LoadData(params, nil)
	require.NoError(t, err)
	require.Equal(t, 23, len(data))
	require.NotNil(t, metaData)
	require.Equal(t, 0, len(dataErrors))

	d := data[0]
	require.Equal(t, 1, d[0].ContinuousFeatures.Rows())
	require.Equal(t, 8, len(d[0].CategoricalFeatures))

	params.DataFile = "../../datasets/breast_cancer/breast-cancer.test"
	testMetaData, data, dataErrors, err := LoadData(params, metaData)
	require.NoError(t, err)
	require.Equal(t, metaData, testMetaData)
	require.Equal(t, 1, len(dataErrors)) // Line 8 contains a category value for Age not present in training
	require.Equal(t, 6, len(data))
}
