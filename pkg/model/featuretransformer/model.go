package featuretransformer

import (
	"math"

	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
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
		Layer1: &Layer{
			InputDimension:               numInputFeatures,
			IntermediateFeatureDimension: featureDimension,
			DenseLayer:                   linear.New(numInputFeatures, 2*featureDimension, linear.BiasGrad(false)),
			BatchNormLayer:               createBatchNormModels(numSteps, featureDimension, batchMomentum),
			NumSteps:                     numSteps,
		},
		Layer2: &Layer{
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
		result[i] = batchnorm.NewWithMomentum(2*featureDimension, mat.Float(batchMomentum))
	}
	return result
}

func (m *Model) Init(generator *rand.LockedRand) {
	m.Layer1.Init(generator)
	m.Layer2.Init(generator)
}

var SquareRootHalf = mat.Float(math.Sqrt(0.5))

func (m *Model) Forward(step int, xs []ag.Node) []ag.Node {
	return m.forward(step, xs, false)
}

func (m *Model) ForwardSkipResidualInput(step int, xs []ag.Node) []ag.Node {
	return m.forward(step, xs, true)
}

func (m *Model) forward(step int, xs []ag.Node, skipResidualInput bool) []ag.Node {
	g := m.Graph()
	theta := g.Constant(SquareRootHalf)

	l1 := m.Layer1.Forward(step, xs)
	if !skipResidualInput {
		for i := range xs {
			l1[i] = g.Mul(g.Add(l1[i], xs[i]), theta)
		}
	}
	l2 := m.Layer2.Forward(step, l1)
	for i := range xs {
		l2[i] = g.Mul(g.Add(l1[i], l2[i]), theta)
	}
	return l2
}
