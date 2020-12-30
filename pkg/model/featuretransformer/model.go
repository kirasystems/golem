package featuretransformer

import (
	"math"

	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/batchnorm"
)

var (
	_ nn.Model = &Model{}
)

// ContinuousFeatures Transformer Block
type Model struct {
	nn.BaseModel
	Layer1 *Layer
	Layer2 *Layer
}

func New(numInputFeatures, featureDimension, numSteps int, batchMomentum float64) *Model {
	return &Model{
		BaseModel: nn.BaseModel{RCS: true},
		Layer1: &Layer{
			BaseModel:                    nn.BaseModel{RCS: true},
			InputDimension:               numInputFeatures,
			IntermediateFeatureDimension: featureDimension,
			DenseLayer:                   linear.New(numInputFeatures, 2*featureDimension, linear.BiasGrad(false)),
			BatchNormLayer:               createBatchNormModels(numSteps, featureDimension, batchMomentum),
			NumSteps:                     numSteps,
		},
		Layer2: &Layer{
			BaseModel:                    nn.BaseModel{RCS: true},
			InputDimension:               featureDimension,
			IntermediateFeatureDimension: featureDimension,
			DenseLayer:                   linear.New(featureDimension, 2*featureDimension, linear.BiasGrad(false)),
			BatchNormLayer:               createBatchNormModels(numSteps, featureDimension, batchMomentum),
			NumSteps:                     numSteps,
		},
	}
}

func createBatchNormModels(steps, featureDimension int, batchMomentum float64) []*batchnorm.Model {
	result := make([]*batchnorm.Model, steps)
	for i := range result {
		result[i] = batchnorm.NewWithMomentum(2*featureDimension, batchMomentum)
	}
	return result
}

func (m *Model) Init(generator *rand.LockedRand) {
	m.Layer1.Init(generator)
	m.Layer2.Init(generator)
}

var SquareRootHalf = math.Sqrt(0.5)

type Input struct {
	Step              int
	Xs                []ag.Node
	SkipResidualInput bool
}

func (m *Model) Forward(in interface{}) interface{} {
	input := in.(Input)
	step := input.Step
	g := m.Graph()
	theta := g.Constant(SquareRootHalf)

	l1 := m.Layer1.Forward(LayerInput{Step: step, Xs: input.Xs}).([]ag.Node)
	if !input.SkipResidualInput {
		for i := range input.Xs {
			l1[i] = g.Mul(g.Add(l1[i], input.Xs[i]), theta)
		}
	}
	l2 := m.Layer2.Forward(LayerInput{Step: step, Xs: l1}).([]ag.Node)
	for i := range input.Xs {
		l2[i] = g.Mul(g.Add(l1[i], l2[i]), theta)
	}
	return l2
}
