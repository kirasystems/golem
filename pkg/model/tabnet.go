package model

import (
	"golem/pkg/model/featuretransformer"

	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/batchnorm"
)

var (
	_ nn.Model = &TabNet{}
)

//  TabNet is an implementation of:
// "TabNet: Attentive Interpretable Tabular Learning" - https://arxiv.org/abs/1908.07442
type TabNet struct {
	nn.BaseModel
	TabNetConfig
	FeatureBatchNorm *batchnorm.Model

	SharedFeatureTransformer     *featuretransformer.Model
	StepFeatureTransformers      []*featuretransformer.Model
	AttentionTransformer         []*linear.Model
	AttentionBatchNorm           []*batchnorm.Model
	OutputLayer                  *linear.Model
	CategoricalFeatureEmbeddings []nn.Param `spago:"type:weights"`
	AttentionEntropy             []ag.Node  `spago:"scope:processor"`

	// AttentionMasks holds, after Forward, the feature attention masks.
	AttentionMasks []AttentionMask `spago:"scope:processor"`
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
		FeatureBatchNorm:             batchnorm.NewWithMomentum(config.NumColumns, mat.Float(config.BatchMomentum)),
		SharedFeatureTransformer:     featuretransformer.New(config.NumColumns, config.IntermediateFeatureDimension, config.NumDecisionSteps, config.BatchMomentum),
		StepFeatureTransformers:      newStepFeatureTransformers(config),
		AttentionTransformer:         createLinearTransformers(config),
		AttentionBatchNorm:           createBatchNormModels(config),
		OutputLayer:                  linear.New(config.IntermediateFeatureDimension, config.OutputDimension, linear.BiasGrad(false)),
		CategoricalFeatureEmbeddings: newCategoricalFeatureEmbeddings(config),
	}
}

func createBatchNormModels(config TabNetConfig) []*batchnorm.Model {
	result := make([]*batchnorm.Model, config.NumDecisionSteps)
	for i := range result {
		result[i] = batchnorm.NewWithMomentum(config.NumColumns, mat.Float(config.BatchMomentum))
	}
	return result

}

func createLinearTransformers(config TabNetConfig) []*linear.Model {
	result := make([]*linear.Model, config.NumDecisionSteps)
	for i := range result {
		result[i] = linear.New(config.IntermediateFeatureDimension, config.NumColumns, linear.BiasGrad(false))
	}
	return result
}

func newCategoricalFeatureEmbeddings(config TabNetConfig) []nn.Param {
	embeddings := make([]nn.Param, config.NumCategoricalEmbeddings)
	for i := range embeddings {
		embeddings[i] = nn.NewParam(mat.NewEmptyVecDense(config.CategoricalEmbeddingDimension), nn.RequiresGrad(true))
	}
	return embeddings
}

func newStepFeatureTransformers(config TabNetConfig) []*featuretransformer.Model {
	stepFeatureTransformers := make([]*featuretransformer.Model, config.NumDecisionSteps)
	for i := range stepFeatureTransformers {
		stepFeatureTransformers[i] = featuretransformer.New(config.IntermediateFeatureDimension, config.IntermediateFeatureDimension, 1, config.BatchMomentum)
	}
	return stepFeatureTransformers
}

func (m *TabNet) Init(generator *rand.LockedRand) {
	m.SharedFeatureTransformer.Init(generator)
	for _, t := range m.StepFeatureTransformers {
		t.Init(generator)
	}
	gain := initializers.Gain(ag.OpIdentity)
	for _, transformer := range m.AttentionTransformer {
		initializers.XavierUniform(transformer.W.Value(), gain, generator)
	}
	initializers.XavierUniform(m.OutputLayer.W.Value(), gain, generator)

	for _, p := range m.CategoricalFeatureEmbeddings {
		initializers.Uniform(p.Value(), -0.1, 0.1, generator)

	}
}

type StepAttentionMask []mat.Float
type AttentionMask []StepAttentionMask

func NewAttentionMask(numSteps, numCols int) AttentionMask {
	mask := make(AttentionMask, numSteps)
	for step := range mask {
		mask[step] = make(StepAttentionMask, numCols)
	}
	return mask
}

func (m *TabNet) Forward(xs []ag.Node) []ag.Node {
	g := m.Graph()

	if m.Mode() == nn.Inference {
		if len(m.AttentionMasks) != len(xs) {
			m.allocateAttentionMasks(len(xs))
		}
	}

	input := m.FeatureBatchNorm.Forward(xs...)

	complementaryAggregatedMaskValues := make([]ag.Node, len(xs))
	for i := range xs {
		complementaryAggregatedMaskValues[i] = g.NewVariable(mat.NewInitVecDense(m.NumColumns, 1.0), true)
	}

	m.AttentionEntropy = make([]ag.Node, len(xs))
	outputAggregated := make([]ag.Node, len(xs))
	maskedFeatures := m.copy(input)

	for i := 0; i < m.NumDecisionSteps; i++ {
		transformed := m.SharedFeatureTransformer.ForwardSkipResidualInput(i, maskedFeatures)
		transformed = m.StepFeatureTransformers[i].Forward(0, transformed)

		if i > 0 {
			decision := make([]ag.Node, len(xs))
			for k := range xs {
				decision[k] = g.ReLU(transformed[k])
				outputAggregated[k] = g.Add(outputAggregated[k], decision[k])
			}
		}

		if i == m.NumDecisionSteps-1 {
			continue // skip attention entropy calculation
		}

		mask := m.AttentionBatchNorm[i].Forward(m.AttentionTransformer[i].Forward(transformed...)...)
		for k := range mask {
			mask[k] = g.Prod(mask[k], complementaryAggregatedMaskValues[k])
			mask[k] = g.SparseMax(mask[k])
			if m.Mode() == nn.Inference {
				copy(m.AttentionMasks[k][i], mask[k].Value().Data())
			}
			complementaryAggregatedMaskValues[k] = g.Prod(complementaryAggregatedMaskValues[k],
				g.Neg(g.SubScalar(mask[k], g.Constant(mat.Float(m.RelaxationFactor)))))
			maskedFeatures[k] = g.Prod(input[k], mask[k])
			stepAttentionEntropy := g.ReduceSum(g.Prod(g.Neg(mask[k]), g.Log(g.AddScalar(mask[k], g.Constant(Epsilon)))))
			stepAttentionEntropy = g.DivScalar(stepAttentionEntropy, g.Constant(mat.Float(float64(m.NumDecisionSteps-1))))
			m.AttentionEntropy[k] = g.Add(m.AttentionEntropy[k], stepAttentionEntropy)
		}
	}

	return m.OutputLayer.Forward(outputAggregated...)
}

// copy makes a copy of input in a gradient-preserving way
func (m *TabNet) copy(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = m.Graph().Identity(x)
	}
	return ys
}

func (m *TabNet) allocateAttentionMasks(len int) {
	m.AttentionMasks = make([]AttentionMask, len)
	for i := range m.AttentionMasks {
		m.AttentionMasks[i] = NewAttentionMask(m.NumDecisionSteps-1, m.NumColumns)
	}
}
