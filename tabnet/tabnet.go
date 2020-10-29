// Package tabnet is an implementation of:
// "Model: Attentive Interpretable Tabular Learning" - https://arxiv.org/abs/1908.07442
package tabnet

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/batchnorm"
	"math"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
	_ nn.Processor = &FeatureTransformerProcessor{}
	_ nn.Processor = &FeatureTransformerBlockProcessor{}
	_ nn.Model     = &FeatureTransformer{}
	_ nn.Model     = &FeatureTransformerBlock{}
)

var SquareRootHalf = math.Sqrt(0.5)

func glu(g *ag.Graph, dim int, x ag.Node) ag.Node {
	half := dim / 2
	value := g.View(x, 0, 0, 0, half)
	gate := g.View(x, 0, half, 0, dim)

	return g.Mul(value, g.Sigmoid(gate))
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
	transformedInput := f.batchNormProcessor.Forward(f.denseLayerProcessor.Forward(xs...)...)
	out := make([]ag.Node, len(xs))
	for i := range out {
		out[i] = glu(f.Graph, f.inputDimension, transformedInput[i])
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
func (f *FeatureTransformerBlock) NewProcNoResidual(g *ag.Graph) nn.Processor {
	out := f.NewProc(g)
	out.(*FeatureTransformerBlockProcessor).skipResidualInput = true
	return out
}

func NewFeatureTransformerBlock(featureDimension int) *FeatureTransformerBlock {

	return &FeatureTransformerBlock{
		Layer1: &FeatureTransformer{
			InputDimension: featureDimension,
			DenseLayer:     linear.New(featureDimension, 2*featureDimension),
			BatchNormLayer: batchnorm.New(featureDimension),
		},
		Layer2: &FeatureTransformer{
			InputDimension: featureDimension,
			DenseLayer:     linear.New(featureDimension, 2*featureDimension),
			BatchNormLayer: batchnorm.New(featureDimension),
		},
	}
}

type Model struct {
	NumDecisionSteps int
	NumColumns       int
	FeatureDimension int
	OutputDimension  int
	RelaxationFactor float64
	BatchMomentum    float64
	VirtualBatchSize int
	Epsilon          float64

	FeatureBatchNorm         *batchnorm.Model
	SharedFeatureTransformer *FeatureTransformerBlock
	StepFeatureTransformers  []*FeatureTransformerBlock

	AttentionTransformer *linear.Model
	AttentionBatchNorm   *batchnorm.Model
	OutputLayer          *linear.Model
}

func NewModel(numDecisionSteps int, numColumns int, featureDimension int, outputDimension int,
	relaxationFactor float64, batchMomentum float64, virtualBatchSize int, epsilon float64) *Model {

	stepFeatureTransformers := make([]*FeatureTransformerBlock, numDecisionSteps)
	for i := range stepFeatureTransformers {
		stepFeatureTransformers[i] = NewFeatureTransformerBlock(featureDimension)
	}

	return &Model{NumDecisionSteps: numDecisionSteps,
		NumColumns:       numColumns,
		FeatureDimension: featureDimension,
		OutputDimension:  outputDimension,
		RelaxationFactor: relaxationFactor,
		BatchMomentum:    batchMomentum,
		VirtualBatchSize: virtualBatchSize,
		Epsilon:          epsilon,

		FeatureBatchNorm:         batchnorm.New(numColumns),
		SharedFeatureTransformer: NewFeatureTransformerBlock(featureDimension),
		StepFeatureTransformers:  stepFeatureTransformers,
		AttentionTransformer:     linear.New(featureDimension, numColumns),
		AttentionBatchNorm:       batchnorm.New(numColumns),
		OutputLayer:              linear.New(featureDimension, outputDimension),
	}
}

type Processor struct {
	nn.BaseProcessor
	model                         *Model
	featureBatchNormProcessor     nn.Processor
	sharedTransformerProcessor    nn.Processor
	stepTransformerProcessors     []nn.Processor
	attentionTransformerProcessor nn.Processor
	attentionBatchNormProcessor   nn.Processor
	outputProcessor               nn.Processor

	attentionEntropy []ag.Node // computed by forward
}

func (m *Model) NewProc(g *ag.Graph) nn.Processor {

	stepTransformerProcessors := make([]nn.Processor, m.NumDecisionSteps)
	for i := range stepTransformerProcessors {
		stepTransformerProcessors[i] = m.StepFeatureTransformers[i].NewProc(g)
	}
	return &Processor{
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

func (p Processor) Forward(xs ...ag.Node) []ag.Node {
	g := p.Graph

	input := p.featureBatchNormProcessor.Forward(xs...)

	outputAggregated := make([]ag.Node, len(xs))
	for i := range xs {
		outputAggregated[i] = g.NewVariable(mat.NewEmptyVecDense(p.model.OutputDimension), true)
	}

	complementaryAggregatedMaskValues := make([]ag.Node, len(xs))
	for i := range xs {
		complementaryAggregatedMaskValues[i] = g.NewVariable(mat.NewInitVecDense(p.model.NumColumns, 1.0), true)
	}

	p.attentionEntropy = make([]ag.Node, len(xs))
	for i := range xs {
		p.attentionEntropy[i] = g.NewVariable(mat.NewScalar(0.0), true)
	}

	//TODO: use_bias=false? (linear)

	for i := 0; i < p.model.NumDecisionSteps; i++ {
		transformed := p.sharedTransformerProcessor.Forward(input...)
		transformed = p.stepTransformerProcessors[i].Forward(transformed...)
		if i > 0 {
			decision := make([]ag.Node, len(xs))
			for i := range xs {
				decision[i] = g.ReLU(transformed[i])
				outputAggregated[i] = g.Add(outputAggregated[i], decision[i])
			}
		}

		if i < p.model.NumDecisionSteps-1 {
			mask := p.attentionBatchNormProcessor.Forward(p.attentionTransformerProcessor.Forward(transformed...)...)
			for i := range mask {
				mask[i] = g.Mul(mask[i], complementaryAggregatedMaskValues[i])
				mask[i] = g.SparseMax(mask[i])
				complementaryAggregatedMaskValues[i] = g.Mul(complementaryAggregatedMaskValues[i],
					g.Sub(g.Constant(p.model.RelaxationFactor), mask[i]))
				input[i] = g.Mul(input[i], mask[i])
				p.attentionEntropy[i] = g.Add(p.attentionEntropy[i],
					g.ReduceSum(g.Mul(g.Neg(mask[i]), g.Log(g.Add(mask[i], g.Constant(p.model.Epsilon))))))
			}

		}
	}

	return outputAggregated
}
