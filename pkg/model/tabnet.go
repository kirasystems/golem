package model

import (
	"math"

	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/batchnorm"
)

var (
	_ nn.Model     = &TabNet{}
	_ nn.Processor = &TabNetProcessor{}
	_ nn.Processor = &FeatureTransformerProcessor{}
	_ nn.Processor = &FeatureTransformerBlockProcessor{}
	_ nn.Model     = &FeatureTransformer{}
	_ nn.Model     = &FeatureTransformerBlock{}
)

var SquareRootHalf = math.Sqrt(0.5)

func glu(g *ag.Graph, dim int, x ag.Node) ag.Node {
	half := dim / 2
	value := g.View(x, 0, 0, half, 1)
	gate := g.View(x, half, 0, half, 1)

	return g.Prod(value, g.Sigmoid(gate))
}

type FeatureTransformer struct {
	InputDimension int
	DenseLayer     *linear.Model
	BatchNormLayer *batchnorm.Model
}

type FeatureTransformerProcessor struct {
	nn.BaseProcessor
	inputDimension      int
	denseLayerProcessor nn.Processor
	batchNormProcessor  nn.Processor
}

func (f *FeatureTransformerProcessor) Forward(xs ...ag.Node) []ag.Node {
	transformedInput := f.denseLayerProcessor.Forward(xs...)
	transformedInput = f.batchNormProcessor.Forward(transformedInput...)
	out := make([]ag.Node, len(xs))
	for i := range out {
		out[i] = glu(f.Graph, 2*f.inputDimension, transformedInput[i])
	}
	return out
}

func (f *FeatureTransformer) NewProc(g *ag.Graph) nn.Processor {
	return &FeatureTransformerProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             f,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		inputDimension:      f.InputDimension,
		denseLayerProcessor: f.DenseLayer.NewProc(g),
		batchNormProcessor:  f.BatchNormLayer.NewProc(g),
	}
}

func (f *FeatureTransformerProcessor) SetMode(mode nn.ProcessingMode) {
	f.Mode = mode
	f.denseLayerProcessor.SetMode(mode)
	f.batchNormProcessor.SetMode(mode)
}

func (f *FeatureTransformer) Init(generator *rand.LockedRand) {
	initializers.XavierUniform(f.DenseLayer.W.Value(), initializers.Gain(ag.OpSigmoid), generator)
}

type FeatureTransformerBlock struct {
	Layer1 *FeatureTransformer
	Layer2 *FeatureTransformer
}

type FeatureTransformerBlockProcessor struct {
	nn.BaseProcessor
	layer1Processor   nn.Processor
	layer2Processor   nn.Processor
	skipResidualInput bool
}

func (f *FeatureTransformerBlockProcessor) Forward(xs ...ag.Node) []ag.Node {

	g := f.Graph
	theta := g.Constant(SquareRootHalf)

	l1 := f.layer1Processor.Forward(xs...)
	if !f.skipResidualInput {
		for i := range xs {
			l1[i] = g.Mul(g.Add(l1[i], xs[i]), theta)
		}
	}
	l2 := f.layer2Processor.Forward(l1...)
	for i := range xs {
		l2[i] = g.Mul(g.Add(l1[i], l2[i]), theta)
	}
	return l2
}

func (f *FeatureTransformerBlock) NewProc(g *ag.Graph) nn.Processor {
	return &FeatureTransformerBlockProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             f,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		layer1Processor: f.Layer1.NewProc(g),
		layer2Processor: f.Layer2.NewProc(g),
	}
}
func (f *FeatureTransformerBlockProcessor) SetMode(mode nn.ProcessingMode) {
	f.Mode = mode
	f.layer1Processor.SetMode(mode)
	f.layer2Processor.SetMode(mode)
}
func (f *FeatureTransformerBlock) NewProcNoResidual(g *ag.Graph) nn.Processor {
	out := f.NewProc(g)
	out.(*FeatureTransformerBlockProcessor).skipResidualInput = true
	return out
}

func (f *FeatureTransformerBlock) Init(generator *rand.LockedRand) {
	f.Layer1.Init(generator)
	f.Layer2.Init(generator)
}
func NewFeatureTransformerBlock(featureDimension int, batchMomentum float64) *FeatureTransformerBlock {
	return &FeatureTransformerBlock{
		Layer1: &FeatureTransformer{
			InputDimension: featureDimension,
			DenseLayer:     linear.New(featureDimension, 2*featureDimension, linear.BiasGrad(false)),
			BatchNormLayer: batchnorm.NewWithMomentum(2*featureDimension, batchMomentum),
		},
		Layer2: &FeatureTransformer{
			InputDimension: featureDimension,
			DenseLayer:     linear.New(featureDimension, 2*featureDimension, linear.BiasGrad(false)),
			BatchNormLayer: batchnorm.NewWithMomentum(2*featureDimension, batchMomentum),
		},
	}
}

//  TabNet is an implementation of:
// "TabNet: Attentive Interpretable Tabular Learning" - https://arxiv.org/abs/1908.07442
type TabNet struct {
	NumDecisionSteps   int
	NumColumns         int
	FeatureDimension   int
	OutputDimension    int
	RelaxationFactor   float64
	BatchMomentum      float64
	VirtualBatchSize   int
	SparsityLossWeight float64

	FeatureBatchNorm         *batchnorm.Model
	SharedFeatureTransformer *FeatureTransformerBlock
	StepFeatureTransformers  []*FeatureTransformerBlock

	AttentionTransformer *linear.Model
	AttentionBatchNorm   *batchnorm.Model
	OutputLayer          *linear.Model
}

const Epsilon = 0.00001

type TabNetParameters struct {
	NumDecisionSteps   int
	NumColumns         int
	FeatureDimension   int
	OutputDimension    int
	RelaxationFactor   float64
	BatchMomentum      float64
	VirtualBatchSize   int
	SparsityLossWeight float64
}

func NewTabNet(p TabNetParameters) *TabNet {

	stepFeatureTransformers := make([]*FeatureTransformerBlock, p.NumDecisionSteps)
	for i := range stepFeatureTransformers {
		stepFeatureTransformers[i] = NewFeatureTransformerBlock(p.FeatureDimension, p.BatchMomentum)
	}

	return &TabNet{NumDecisionSteps: p.NumDecisionSteps,
		NumColumns:         p.NumColumns,
		FeatureDimension:   p.FeatureDimension,
		OutputDimension:    p.OutputDimension,
		RelaxationFactor:   p.RelaxationFactor,
		BatchMomentum:      p.BatchMomentum,
		VirtualBatchSize:   p.VirtualBatchSize,
		SparsityLossWeight: p.SparsityLossWeight,

		FeatureBatchNorm:         batchnorm.NewWithMomentum(p.NumColumns, p.BatchMomentum),
		SharedFeatureTransformer: NewFeatureTransformerBlock(p.FeatureDimension, p.BatchMomentum),
		StepFeatureTransformers:  stepFeatureTransformers,
		AttentionTransformer:     linear.New(p.FeatureDimension, p.NumColumns, linear.BiasGrad(false)),
		AttentionBatchNorm:       batchnorm.NewWithMomentum(p.NumColumns, p.BatchMomentum),
		OutputLayer:              linear.New(p.FeatureDimension, p.OutputDimension, linear.BiasGrad(false)),
	}
}

type TabNetProcessor struct {
	nn.BaseProcessor
	model                         *TabNet
	featureBatchNormProcessor     nn.Processor
	sharedTransformerProcessor    nn.Processor
	stepTransformerProcessors     []nn.Processor
	attentionTransformerProcessor nn.Processor
	attentionBatchNormProcessor   nn.Processor
	outputProcessor               nn.Processor

	AttentionEntropy []ag.Node // computed by forward
}

func (m *TabNet) NewProc(g *ag.Graph) nn.Processor {

	stepTransformerProcessors := make([]nn.Processor, m.NumDecisionSteps)
	for i := range stepTransformerProcessors {
		stepTransformerProcessors[i] = m.StepFeatureTransformers[i].NewProc(g)
	}
	return &TabNetProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		model:                         m,
		featureBatchNormProcessor:     m.FeatureBatchNorm.NewProc(g),
		sharedTransformerProcessor:    m.SharedFeatureTransformer.NewProcNoResidual(g),
		stepTransformerProcessors:     stepTransformerProcessors,
		outputProcessor:               m.OutputLayer.NewProc(g),
		attentionTransformerProcessor: m.AttentionTransformer.NewProc(g),
		attentionBatchNormProcessor:   m.AttentionBatchNorm.NewProc(g),
	}
}

func (p *TabNetProcessor) SetMode(mode nn.ProcessingMode) {
	p.Mode = mode
	p.featureBatchNormProcessor.SetMode(mode)
	p.sharedTransformerProcessor.SetMode(mode)
	nn.SetProcessingMode(mode, p.stepTransformerProcessors...)
	p.outputProcessor.SetMode(mode)
	p.attentionTransformerProcessor.SetMode(mode)
	p.attentionBatchNormProcessor.SetMode(mode)

}

func (p *TabNetProcessor) Forward(xs ...ag.Node) []ag.Node {
	g := p.Graph

	input := p.featureBatchNormProcessor.Forward(xs...)

	complementaryAggregatedMaskValues := make([]ag.Node, len(xs))
	for i := range xs {
		complementaryAggregatedMaskValues[i] = g.NewVariable(mat.NewInitVecDense(p.model.NumColumns, 1.0), true)
	}

	p.AttentionEntropy = make([]ag.Node, len(xs))
	outputAggregated := make([]ag.Node, len(xs))
	maskedFeatures := p.copy(input)

	for i := 0; i < p.model.NumDecisionSteps; i++ {
		transformed := p.sharedTransformerProcessor.Forward(maskedFeatures...)
		transformed = p.stepTransformerProcessors[i].Forward(transformed...)
		if i > 0 {
			decision := make([]ag.Node, len(xs))
			for k := range xs {
				decision[k] = g.ReLU(transformed[k])
				outputAggregated[k] = g.Add(outputAggregated[k], decision[k])
			}
		}

		if i < p.model.NumDecisionSteps-1 {
			mask := p.attentionBatchNormProcessor.Forward(p.attentionTransformerProcessor.Forward(transformed...)...)
			for k := range mask {
				mask[k] = g.Prod(mask[k], complementaryAggregatedMaskValues[k])
				mask[k] = g.SparseMax(mask[k])
				complementaryAggregatedMaskValues[k] = g.Prod(complementaryAggregatedMaskValues[k],
					g.Neg(g.SubScalar(mask[k], g.Constant(p.model.RelaxationFactor))))
				maskedFeatures[k] = g.Prod(input[k], mask[k])
				stepAttentionEntropy := g.ReduceSum(g.Prod(g.Neg(mask[k]), g.Log(g.AddScalar(mask[k], g.Constant(Epsilon)))))
				stepAttentionEntropy = g.DivScalar(stepAttentionEntropy, g.Constant(float64(p.model.NumDecisionSteps-1)))
				p.AttentionEntropy[k] = g.Add(p.AttentionEntropy[k],
					stepAttentionEntropy)
			}
		}
	}

	return p.outputProcessor.Forward(outputAggregated...)
}

// copy makes a copy of input in a gradient-preserving way
func (p *TabNetProcessor) copy(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = p.Graph.Identity(x)
	}
	return ys
}

func (m *TabNet) Init(generator *rand.LockedRand) {
	m.SharedFeatureTransformer.Init(generator)
	for _, t := range m.StepFeatureTransformers {
		t.Init(generator)
	}
	gain := initializers.Gain(ag.OpIdentity)
	initializers.XavierUniform(m.AttentionTransformer.W.Value(), gain, generator)
	initializers.XavierUniform(m.OutputLayer.W.Value(), gain, generator)
}
