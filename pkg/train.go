package pkg

import (
	mathrand "math/rand"

	mat "github.com/nlpodyssey/spago/pkg/mat32"

	"golem/pkg/io"
	"golem/pkg/model"

	"github.com/rs/zerolog/log"

	"os"

	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/losses"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd/adam"
)

type TrainingParameters struct {
	BatchSize          int
	NumEpochs          int
	LearningRate       float64
	ReportInterval     int
	RndSeed            uint64
	CategoricalColumns []string
}

type lossFunc func(g *ag.Graph, prediction ag.Node, target interface{}) ag.Node

func crossEntropyLoss(g *ag.Graph, prediction ag.Node, target interface{}) ag.Node {
	return losses.CrossEntropy(g, prediction, int(target.(mat.Float)))
}

func mseLoss(g *ag.Graph, prediction ag.Node, target interface{}) ag.Node {
	return losses.MSE(g, prediction, g.NewScalar(target.(mat.Float)), false)
}

type dataPreProcessor interface {
	process(g *ag.Graph, input []ag.Node) []ag.Node
}

type inputDropoutPreprocessor struct {
	P              mat.Float
	Rand           *rand.LockedRand
	CurrentMasks   []mat.Matrix
	InputDimension int
}

func NewDropoutPreprocessor(p mat.Float, r *rand.LockedRand, inputDimension, batchSize int) *inputDropoutPreprocessor {
	dropout := &inputDropoutPreprocessor{
		P:              p,
		Rand:           r,
		CurrentMasks:   make([]mat.Matrix, batchSize),
		InputDimension: inputDimension,
	}
	for i := range dropout.CurrentMasks {
		dropout.CurrentMasks[i] = mat.NewEmptyVecDense(inputDimension)
	}
	return dropout
}

func (d *inputDropoutPreprocessor) process(g *ag.Graph, input []ag.Node) []ag.Node {
	result := make([]ag.Node, len(input))
	for i := range input {
		for j := 0; j < d.InputDimension; j++ {
			r := d.Rand.Float()
			if r <= d.P {
				r = 1.0
			} else {
				r = 0.0
			}
			d.CurrentMasks[i].Set(0, j, r)
		}
		result[i] = g.Prod(input[i], g.NewVariable(d.CurrentMasks[i], false))
	}
	return result

}

func lossFor(metadata *model.Metadata) lossFunc {
	switch metadata.TargetType() {
	case model.Continuous:
		return mseLoss
	case model.Categorical:
		return crossEntropyLoss
	default:
		log.Panic().Msgf("unsupported model type received: %d", metadata.TargetType())
		return nil
	}
}

type Trainer struct {
	params       TrainingParameters
	optimizer    *gd.GradientDescent
	model        *model.TabNet
	lossFunc     lossFunc
	preProcessor dataPreProcessor
}

func Train(trainFile, outputFileName, targetColumn string, config model.TabNetConfig, trainingParams TrainingParameters) {
	t := &Trainer{params: trainingParams}

	rndGen := rand.NewLockedRand(trainingParams.RndSeed)

	metaData, dataSet, dataErrors, err := io.LoadData(io.DataParameters{
		DataFile:           trainFile,
		TargetColumn:       targetColumn,
		CategoricalColumns: io.NewSet(trainingParams.CategoricalColumns...),
		BatchSize:          trainingParams.BatchSize}, nil)

	if err != nil {
		log.Fatal().Msgf("Error reading training data: %s", err)
		return
	}
	printDataErrors(dataErrors)
	if len(dataSet.Data) == 0 {
		log.Fatal().Msgf("No data to train")
		return
	}
	dataSet.Rand = mathrand.New(mathrand.NewSource(int64(trainingParams.RndSeed)))

	//Overwrite values that are  only known after parsing the dataset
	config.NumColumns = metaData.FeatureCount()
	config.NumCategoricalEmbeddings = len(metaData.CategoricalValuesMap.ValueToIndex)
	switch metaData.TargetType() {
	case model.Categorical:
		config.OutputDimension = metaData.TargetMap.Size()
	case model.Continuous:
		config.OutputDimension = 1
	}

	t.model = model.NewTabNet(config)
	t.lossFunc = lossFor(metaData)
	t.model.Init(rndGen)

	updaterConfig := adam.NewDefaultConfig() // TODO: `radam` may provide better results
	updaterConfig.StepSize = mat.Float(trainingParams.LearningRate)
	updater := adam.New(updaterConfig)
	const GradientClipThreshold = 2000.0 // TODO: get from configuration
	t.optimizer = gd.NewOptimizer(updater, nn.NewDefaultParamsIterator(t.model),
		gd.ClipGradByValue(GradientClipThreshold),
		gd.ConcurrentComputations(1))

	for epoch := 0; epoch < trainingParams.NumEpochs; epoch++ {
		dataSet.ResetOrder(io.RandomOrder)
		t.optimizer.IncEpoch()
		i := 0
		for batch := dataSet.Next(); len(batch) > 0; batch = dataSet.Next() {
			totalLoss, targetLoss, sparsityLoss := t.trainBatch(batch)
			t.optimizer.Optimize()
			if i%t.params.ReportInterval == 0 {
				log.Info().Int("epoch", epoch).Int("batch", i).
					Float32("totalLoss", totalLoss).
					Float32("targetLoss", targetLoss).
					Float32("sparsityLoss", sparsityLoss).Msgf("")
			}
			i++
		}
	}

	m := model.Model{
		MetaData: metaData,
		TabNet:   t.model,
	}

	outputFile, err := os.Create(outputFileName)
	if err != nil {
		log.Printf("Error creating output file %s: %s", outputFileName, err)
	}
	defer outputFile.Close()

	err = io.SaveModel(&m, outputFile)
	if err != nil {
		log.Printf("Error saving model to %s: %s", outputFileName, err)
	}

	err = testInternal(&m, dataSet, "", "")
	if err != nil {
		log.Fatal().Msg(err.Error())

	}

}

func (t *Trainer) trainBatch(batch io.DataBatch) (mat.Float, mat.Float, mat.Float) {
	t.optimizer.IncBatch()

	g := ag.NewGraph(
		ag.Rand(rand.NewLockedRand(t.params.RndSeed)),
		ag.ConcurrentComputations(1)) // TODO: we might use the same random generator among the batches until we run them concurrently
	defer g.Clear()

	input := createInputNodes(batch, g, t.model)
	if t.preProcessor != nil {
		input = t.preProcessor.process(g, input)
	}
	ctx := nn.Context{Graph: g, Mode: nn.Training}
	modelProc := nn.Reify(ctx, t.model).(*model.TabNet)
	prediction := modelProc.Forward(input)

	var batchLoss, batchTargetLoss, batchSparsityLoss ag.Node
	for i := range batch {
		targetLoss := t.lossFunc(g, prediction[i], batch[i].Target)
		batchTargetLoss = g.Add(batchTargetLoss, targetLoss)
		batchSparsityLoss = g.Add(batchSparsityLoss, modelProc.AttentionEntropy[i])
		exampleAttentionEntropy := g.Mul(modelProc.AttentionEntropy[i], g.Constant(mat.Float(t.model.SparsityLossWeight)))
		exampleLoss := g.Add(targetLoss, exampleAttentionEntropy)
		batchLoss = g.Add(batchLoss, exampleLoss)
	}
	batchSize := g.NewScalar(mat.Float(len(batch)))
	batchLoss = g.Div(batchLoss, batchSize)
	batchTargetLoss = g.Div(batchTargetLoss, batchSize)
	batchSparsityLoss = g.Div(batchSparsityLoss, batchSize)

	g.Backward(batchLoss)
	return batchLoss.ScalarValue(), batchTargetLoss.ScalarValue(), batchSparsityLoss.ScalarValue()
}

func createInputNodes(batch io.DataBatch, g *ag.Graph, model *model.TabNet) []ag.Node {
	input := make([]ag.Node, len(batch))
	for i := range input {
		if batch[i].ContinuousFeatures.Size() > 0 {
			input[i] = g.NewVariable(batch[i].ContinuousFeatures, false)
		}
		numCategoricalFeatures := len(batch[i].CategoricalFeatures)
		if numCategoricalFeatures > 0 {
			featureNodes := make([]ag.Node, 0, numCategoricalFeatures+1)
			if input[i] != nil {
				featureNodes = append(featureNodes, input[i])
			}
			for _, index := range batch[i].CategoricalFeatures {
				featureNodes = append(featureNodes, g.NewWrap(model.CategoricalFeatureEmbeddings[index]))
			}
			input[i] = g.Concat(featureNodes...)
		}
	}
	return input
}
