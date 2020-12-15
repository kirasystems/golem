package model

import (
	"golem/pkg/model/featuretransformer"

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
)

//  TabNet is an implementation of:
// "TabNet: Attentive Interpretable Tabular Learning" - https://arxiv.org/abs/1908.07442
type TabNet struct {
	TabNetConfig
	FeatureBatchNorm             *batchnorm.Model
	SharedFeatureTransformer     *featuretransformer.Model
	StepFeatureTransformers      []*featuretransformer.Model
	AttentionTransformer         *linear.Model
	AttentionBatchNorm           *batchnorm.Model
	OutputLayer                  *linear.Model
	CategoricalFeatureEmbeddings []*nn.Param
}

const Epsilon = 0.00001

type TabNetConfig struct {
	NumDecisionSteps              int
	NumColumns                    int
	IntermediateFeatureDimension  int
	OutputDimension               int
	CategoricalEmbeddingDimension int
	NumCategoricalEmbeddings      int
	RelaxationFactor              float64
	BatchMomentum                 float64
	VirtualBatchSize              int
	SparsityLossWeight            float64
}

func NewTabNet(config TabNetConfig) *TabNet {

	return &TabNet{
		TabNetConfig:                 config,
		FeatureBatchNorm:             batchnorm.NewWithMomentum(config.NumColumns, config.BatchMomentum),
		SharedFeatureTransformer:     featuretransformer.New(config.NumColumns, config.IntermediateFeatureDimension, config.BatchMomentum),
		StepFeatureTransformers:      newStepFeatureTransformers(config),
		AttentionTransformer:         linear.New(config.IntermediateFeatureDimension, config.NumColumns, linear.BiasGrad(false)),
		AttentionBatchNorm:           batchnorm.NewWithMomentum(config.NumColumns, config.BatchMomentum),
		OutputLayer:                  linear.New(config.IntermediateFeatureDimension, config.OutputDimension, linear.BiasGrad(false)),
		CategoricalFeatureEmbeddings: newCategoricalFeatureEmbeddings(config),
	}
}

func newCategoricalFeatureEmbeddings(config TabNetConfig) []*nn.Param {
	embeddings := make([]*nn.Param, config.NumCategoricalEmbeddings)
	for i := range embeddings {
		embeddings[i] = nn.NewParam(mat.NewEmptyVecDense(config.CategoricalEmbeddingDimension), nn.RequiresGrad(true))
	}
	return embeddings
}

func newStepFeatureTransformers(config TabNetConfig) []*featuretransformer.Model {
	stepFeatureTransformers := make([]*featuretransformer.Model, config.NumDecisionSteps)
	for i := range stepFeatureTransformers {
		stepFeatureTransformers[i] = featuretransformer.New(config.IntermediateFeatureDimension, config.IntermediateFeatureDimension, config.BatchMomentum)
	}
	return stepFeatureTransformers
}
func (m *TabNet) Init(generator *rand.LockedRand) {
	m.SharedFeatureTransformer.Init(generator)
	for _, t := range m.StepFeatureTransformers {
		t.Init(generator)
	}
	gain := initializers.Gain(ag.OpIdentity)
	initializers.XavierUniform(m.AttentionTransformer.W.Value(), gain, generator)
	initializers.XavierUniform(m.OutputLayer.W.Value(), gain, generator)

	for _, p := range m.CategoricalFeatureEmbeddings {
		initializers.Uniform(p.Value(), 0.1, 0.1, generator)
	}
}

type TabNetProcessor struct {
	nn.BaseProcessor
	model                         *TabNet
	featureBatchNormProcessor     *batchnorm.Processor
	sharedTransformerProcessor    *featuretransformer.Processor
	stepTransformerProcessors     []*featuretransformer.Processor
	attentionTransformerProcessor *linear.Processor
	attentionBatchNormProcessor   *batchnorm.Processor
	outputProcessor               *linear.Processor

	AttentionEntropy []ag.Node // computed by forward
}

func (m *TabNet) NewProc(ctx nn.Context) nn.Processor {
	stepTransformerProcessors := make([]*featuretransformer.Processor, m.NumDecisionSteps)
	for i := range stepTransformerProcessors {
		stepTransformerProcessors[i] = m.StepFeatureTransformers[i].NewProc(ctx).(*featuretransformer.Processor)
	}
	return &TabNetProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		model:                         m,
		featureBatchNormProcessor:     m.FeatureBatchNorm.NewProc(ctx).(*batchnorm.Processor),
		sharedTransformerProcessor:    m.SharedFeatureTransformer.NewProcNoResidual(ctx).(*featuretransformer.Processor),
		stepTransformerProcessors:     stepTransformerProcessors,
		outputProcessor:               m.OutputLayer.NewProc(ctx).(*linear.Processor),
		attentionTransformerProcessor: m.AttentionTransformer.NewProc(ctx).(*linear.Processor),
		attentionBatchNormProcessor:   m.AttentionBatchNorm.NewProc(ctx).(*batchnorm.Processor),
	}
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
