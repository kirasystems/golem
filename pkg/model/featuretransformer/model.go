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
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// ContinuousFeatures Transformer Block
type Model struct {
	Layer1 *Layer
	Layer2 *Layer
}

func New(featureDimension int, batchMomentum float64) *Model {
	return &Model{
		Layer1: &Layer{
			InputDimension: featureDimension,
			DenseLayer:     linear.New(featureDimension, 2*featureDimension, linear.BiasGrad(false)),
			BatchNormLayer: batchnorm.NewWithMomentum(2*featureDimension, batchMomentum),
		},
		Layer2: &Layer{
			InputDimension: featureDimension,
			DenseLayer:     linear.New(featureDimension, 2*featureDimension, linear.BiasGrad(false)),
			BatchNormLayer: batchnorm.NewWithMomentum(2*featureDimension, batchMomentum),
		},
	}
}

func (f *Model) Init(generator *rand.LockedRand) {
	f.Layer1.Init(generator)
	f.Layer2.Init(generator)
}

type Processor struct {
	nn.BaseProcessor
	layer1Processor   *LayerProcessor
	layer2Processor   *LayerProcessor
	skipResidualInput bool
}

var SquareRootHalf = math.Sqrt(0.5)

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	g := p.Graph
	theta := g.Constant(SquareRootHalf)

	l1 := p.layer1Processor.Forward(xs...)
	if !p.skipResidualInput {
		for i := range xs {
			l1[i] = g.Mul(g.Add(l1[i], xs[i]), theta)
		}
	}
	l2 := p.layer2Processor.Forward(l1...)
	for i := range xs {
		l2[i] = g.Mul(g.Add(l1[i], l2[i]), theta)
	}
	return l2
}

func (f *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             f,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		layer1Processor: f.Layer1.NewProc(ctx).(*LayerProcessor),
		layer2Processor: f.Layer2.NewProc(ctx).(*LayerProcessor),
	}
}

func (f *Model) NewProcNoResidual(ctx nn.Context) nn.Processor {
	out := f.NewProc(ctx)
	out.(*Processor).skipResidualInput = true
	return out
}
