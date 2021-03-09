package model

import (
	"testing"

	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/stretchr/testify/require"
)

const testBatchSize = 20

func TestTabNet_Forward_Decoder(t *testing.T) {

	tests := []struct {
		numColumns      int
		outputDimension int
	}{
		{
			numColumns:      19,
			outputDimension: 1,
		},
		{
			numColumns:      20,
			outputDimension: 2,
		},
	}

	for _, tt := range tests {
		model := NewTabNet(TabNetConfig{
			NumDecisionSteps:              4,
			NumColumns:                    tt.numColumns,
			IntermediateFeatureDimension:  4,
			OutputDimension:               tt.outputDimension,
			CategoricalEmbeddingDimension: 1,
			NumCategoricalEmbeddings:      10,
			RelaxationFactor:              1.5,
			BatchMomentum:                 0.9,
			VirtualBatchSize:              16,
			SparsityLossWeight:            0.001,
		})

		g := ag.NewGraph(ag.Rand(rand.NewLockedRand(42)))
		ctx := nn.Context{Graph: g, Mode: nn.Inference}
		proc := nn.Reify(ctx, model).(*TabNet)
		input := createInput(g, model.TabNetConfig)

		result := proc.Forward(input)
		require.NotNil(t, result)
		require.Equal(t, testBatchSize, len(result.Output))
		for i, r := range result.Output {
			require.Equal(t, tt.outputDimension, r.Value().Rows())
			require.Equal(t, tt.numColumns, result.DecoderOutput[i].Value().Rows())
		}
	}

}

func createInput(g *ag.Graph, config TabNetConfig) []ag.Node {
	input := make([]ag.Node, testBatchSize)
	for i := range input {
		input[i] = g.NewVariable(mat.NewEmptyVecDense(config.NumColumns), false)
	}
	return input

}
