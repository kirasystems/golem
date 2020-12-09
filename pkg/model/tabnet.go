package model

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/batchnorm"
	"golem/pkg/model/featuretransformer"
)

var (
	_ nn.Model     = &TabNet{}
	_ nn.Processor = &TabNetProcessor{}
)

//  TabNet is an implementation of:
// "TabNet: Attentive Interpretable Tabular Learning" - https://arxiv.org/abs/1908.07442
type TabNet struct {
	TabNetConfig
	FeatureBatchNorm         *batchnorm.Model
	SharedFeatureTransformer *featuretransformer.Model
	StepFeatureTransformers  []*featuretransformer.Model
	AttentionTransformer     *linear.Model
	AttentionBatchNorm       *batchnorm.Model
	OutputLayer              *linear.Model
}

const Epsilon = 0.00001

type TabNetConfig struct {
	NumDecisionSteps   int
	NumColumns         int
	FeatureDimension   int
	OutputDimension    int
	RelaxationFactor   float64
	BatchMomentum      float64
	VirtualBatchSize   int
	SparsityLossWeight float64
}

func NewTabNet(config TabNetConfig) *TabNet {
	stepFeatureTransformers := make([]*featuretransformer.Model, config.NumDecisionSteps)
	for i := range stepFeatureTransformers {
		stepFeatureTransformers[i] = featuretransformer.New(config.FeatureDimension, config.BatchMomentum)
	}
	return &TabNet{
		TabNetConfig:             config,
		FeatureBatchNorm:         batchnorm.NewWithMomentum(config.NumColumns, config.BatchMomentum),
		SharedFeatureTransformer: featuretransformer.New(config.FeatureDimension, config.BatchMomentum),
		StepFeatureTransformers:  stepFeatureTransformers,
		AttentionTransformer:     linear.New(config.FeatureDimension, config.NumColumns, linear.BiasGrad(false)),
		AttentionBatchNorm:       batchnorm.NewWithMomentum(config.NumColumns, config.BatchMomentum),
		OutputLayer:              linear.New(config.FeatureDimension, config.OutputDimension, linear.BiasGrad(false)),
	}
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

		if i == p.model.NumDecisionSteps-1 {
			continue // skip attention entropy calculation
		}

		mask := p.attentionBatchNormProcessor.Forward(p.attentionTransformerProcessor.Forward(transformed...)...)
		for k := range mask {
			mask[k] = g.Prod(mask[k], complementaryAggregatedMaskValues[k])
			mask[k] = g.SparseMax(mask[k])
			complementaryAggregatedMaskValues[k] = g.Prod(complementaryAggregatedMaskValues[k],
				g.Neg(g.SubScalar(mask[k], g.Constant(p.model.RelaxationFactor))))
			maskedFeatures[k] = g.Prod(input[k], mask[k])
			stepAttentionEntropy := g.ReduceSum(g.Prod(g.Neg(mask[k]), g.Log(g.AddScalar(mask[k], g.Constant(Epsilon)))))
			stepAttentionEntropy = g.DivScalar(stepAttentionEntropy, g.Constant(float64(p.model.NumDecisionSteps-1)))
			p.AttentionEntropy[k] = g.Add(p.AttentionEntropy[k], stepAttentionEntropy)
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
