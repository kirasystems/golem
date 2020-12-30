package featuretransformer

import (
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/batchnorm"
)

var (
	_ nn.Model = &Layer{}
)

type Layer struct {
	nn.BaseModel
	InputDimension               int
	IntermediateFeatureDimension int
	NumSteps                     int
	DenseLayer                   *linear.Model
	BatchNormLayer               []*batchnorm.Model
}

type LayerInput struct {
	Step int
	Xs   []ag.Node
}

func (m *Layer) Init(generator *rand.LockedRand) {
	initializers.XavierUniform(m.DenseLayer.W.Value(), initializers.Gain(ag.OpSigmoid), generator)
}

func (m *Layer) Forward(in interface{}) interface{} {
	input := in.(LayerInput)
	transformedInput := m.DenseLayer.Forward(input.Xs)
	transformedInput = m.BatchNormLayer[input.Step].Forward(transformedInput)
	out := make([]ag.Node, len(input.Xs))
	for i := range out {
		out[i] = glu(m.Graph(), 2*m.IntermediateFeatureDimension, transformedInput.([]ag.Node)[i])
	}
	return out
}

func glu(g *ag.Graph, dim int, x ag.Node) ag.Node {
	half := dim / 2
	value := g.View(x, 0, 0, half, 1)
	gate := g.View(x, half, 0, half, 1)
	return g.Prod(value, g.Sigmoid(gate))
}
