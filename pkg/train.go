package pkg

import (
	"golem/pkg/io"
	"golem/pkg/model"
	"log"
	"os"

	"github.com/nlpodyssey/spago/pkg/mat/rand"
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

type lossFunc func(g *ag.Graph, prediction ag.Node, target float64) ag.Node

func crossEntropyLoss(g *ag.Graph, prediction ag.Node, target float64) ag.Node {
	return losses.CrossEntropy(g, prediction, int(target))
}

func mseLoss(g *ag.Graph, prediction ag.Node, target float64) ag.Node {
	return losses.MSE(g, prediction, g.NewScalar(target), false)
}

func lossFor(metadata *model.Metadata) lossFunc {
	modelType := metadata.Columns[metadata.TargetColumn].Type
	switch modelType {
	case model.Continuous:
		return mseLoss
	case model.Categorical:
		return crossEntropyLoss
	default:
		log.Panicf("unsupported model type received: %d", modelType)
		return nil
	}
}

type Trainer struct {
	params    TrainingParameters
	optimizer *gd.GradientDescent
	model     *model.TabNet
	lossFunc  lossFunc
}

func Train(trainFile, outputFileName, targetColumn string, config model.TabNetConfig, trainingParams TrainingParameters) {
	t := &Trainer{params: trainingParams}

	rndGen := rand.NewLockedRand(trainingParams.RndSeed)

	metaData, data, dataErrors, err := io.LoadData(io.DataParameters{
		DataFile:           trainFile,
		TargetColumn:       targetColumn,
		CategoricalColumns: io.NewSet(trainingParams.CategoricalColumns...),
		BatchSize:          trainingParams.BatchSize}, nil)

	if err != nil {
		log.Fatalf("Error reading training data: %s", err)
		return
	}
	printDataErrors(dataErrors)
	if len(data) == 0 {
		log.Fatalf("No data to train")
	}

	//Overwrite values that are  only known after parsing the dataset
	config.NumColumns = metaData.FeatureCount()
	config.NumCategoricalEmbeddings = len(metaData.CategoricalValuesMap.ValueToIndex)
	switch metaData.Columns[metaData.TargetColumn].Type {
	case model.Categorical:
		config.OutputDimension = metaData.TargetMap.Size()
	case model.Continuous:
		config.OutputDimension = 1
	}

	t.model = model.NewTabNet(config)
	t.lossFunc = lossFor(metaData)
	t.model.Init(rndGen)

	updaterConfig := adam.NewDefaultConfig() // TODO: `radam` may provide better results
	updaterConfig.StepSize = trainingParams.LearningRate
	updater := adam.New(updaterConfig)
	const GradientClipThreshold = 2000.0 // TODO: get from configuration
	t.optimizer = gd.NewOptimizer(updater, nn.NewDefaultParamsIterator(t.model),
		gd.ClipGradByValue(GradientClipThreshold))

	for epoch := 0; epoch < trainingParams.NumEpochs; epoch++ {
		t.optimizer.IncEpoch()
		for i, batch := range data {
			totalLoss, targetLoss, sparsityLoss := t.trainBatch(batch)
			t.optimizer.Optimize()
			if i%t.params.ReportInterval == 0 {
				log.Printf("Epoch %d batch %d loss %.5f | %.5f | %.5f \n", epoch, i, totalLoss, targetLoss, sparsityLoss)
			}
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

	err = testInternal(&m, data, "")
	if err != nil {
		log.Fatalf(err.Error())
	}

}

func (t *Trainer) trainBatch(batch io.DataBatch) (float64, float64, float64) {
	t.optimizer.IncBatch()

	g := ag.NewGraph(ag.Rand(rand.NewLockedRand(t.params.RndSeed))) // TODO: we might use the same random generator among the batches until we run them concurrently
	defer g.Clear()

	input := createInputNodes(batch, g, t.model)
	ctx := nn.Context{Graph: g, Mode: nn.Training}
	modelProc := nn.Reify(ctx, t.model).(*model.TabNet)
	prediction := modelProc.Forward(input)

	var batchLoss, batchTargetLoss, batchSparsityLoss ag.Node
	for i := range batch {
		targetLoss := t.lossFunc(g, prediction[i], batch[i].Target)
		batchTargetLoss = g.Add(batchTargetLoss, targetLoss)
		batchSparsityLoss = g.Add(batchSparsityLoss, modelProc.AttentionEntropy[i])
		exampleAttentionEntropy := g.Mul(modelProc.AttentionEntropy[i], g.Constant(t.model.SparsityLossWeight))
		exampleLoss := g.Add(targetLoss, exampleAttentionEntropy)
		batchLoss = g.Add(batchLoss, exampleLoss)
	}
	batchSize := g.NewScalar(float64(len(batch)))
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
