package decoder

import (
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"

	"golem/pkg/model/featuretransformer"
)

// Model implements a TabNet decoder
type Model struct {
	nn.BaseModel

	FeatureTransformer *featuretransformer.Model
	DenseLayer         *linear.Model
}

func (d *Model) Forward(xs []ag.Node) []ag.Node {
	transformed := d.FeatureTransformer.Forward(0, xs)
	return d.DenseLayer.Forward(transformed...)
}

func (d *Model) Init(generator *rand.LockedRand) {
	initializers.XavierUniform(d.DenseLayer.W.Value(), initializers.Gain(ag.OpSigmoid), generator)
	d.FeatureTransformer.Init(generator)
}

func NewDecoder(inputDimension, featureDimension, outputDimension int, batchMomentum float64) *Model {
	return &Model{
		FeatureTransformer: featuretransformer.New(inputDimension, featureDimension, 1, batchMomentum),
		DenseLayer:         linear.New(featureDimension, outputDimension, linear.BiasGrad(false)),
	}
}
