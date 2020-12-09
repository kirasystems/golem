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
	_ nn.Model     = &Layer{}
	_ nn.Processor = &LayerProcessor{}
)

type Layer struct {
	InputDimension int
	DenseLayer     *linear.Model
	BatchNormLayer *batchnorm.Model
}

func (m *Layer) Init(generator *rand.LockedRand) {
	initializers.XavierUniform(m.DenseLayer.W.Value(), initializers.Gain(ag.OpSigmoid), generator)
}

type LayerProcessor struct {
	nn.BaseProcessor
	inputDimension      int
	denseLayerProcessor nn.Processor
	batchNormProcessor  nn.Processor
}

func (m *Layer) NewProc(g *ag.Graph) nn.Processor {
	return &LayerProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		inputDimension:      m.InputDimension,
		denseLayerProcessor: m.DenseLayer.NewProc(g),
		batchNormProcessor:  m.BatchNormLayer.NewProc(g),
	}
}

func (p *LayerProcessor) SetMode(mode nn.ProcessingMode) {
	p.Mode = mode
	p.denseLayerProcessor.SetMode(mode)
	p.batchNormProcessor.SetMode(mode)
}

func (p *LayerProcessor) Forward(xs ...ag.Node) []ag.Node {
	transformedInput := p.denseLayerProcessor.Forward(xs...)
	transformedInput = p.batchNormProcessor.Forward(transformedInput...)
	out := make([]ag.Node, len(xs))
	for i := range out {
		out[i] = glu(p.Graph, 2*p.inputDimension, transformedInput[i])
	}
	return out
}

func glu(g *ag.Graph, dim int, x ag.Node) ag.Node {
	half := dim / 2
	value := g.View(x, 0, 0, half, 1)
	gate := g.View(x, half, 0, half, 1)
	return g.Prod(value, g.Sigmoid(gate))
}
