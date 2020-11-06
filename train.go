package main

import (
	"golem/io"
	"golem/model"
	"log"
	"os"

	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/losses"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd/adam"
)

type TrainingParameters struct {
	BatchSize      int
	NumEpochs      int
	LearningRate   float64
	ReportInterval int
	RndSeed        uint64
}

type Trainer struct {
	params    TrainingParameters
	optimizer *gd.GradientDescent
	model     *model.TabNet
}

func Train(trainFile, outputFileName, targetColumn string, params model.TabNetParameters, trainingParams TrainingParameters) {
	t := &Trainer{params: trainingParams}

	rndGen := rand.NewLockedRand(trainingParams.RndSeed)

	metaData, data, err := io.LoadData(trainFile, targetColumn, trainingParams.BatchSize)
	if err != nil {
		log.Fatalf("Error reading training data: %s", err)
		return
	}

	//Overwrite number of feature columns as this is only known after parsing the dataset
	params.NumColumns = metaData.FeatureCount()
	t.model = model.NewTabNet(params)
	t.model.Init(rndGen)

	updaterConfig := adam.NewDefaultConfig()
	updaterConfig.StepSize = trainingParams.LearningRate
	updater := adam.New(updaterConfig)
	const GradientClipTreshold = 2000.0
	t.optimizer = gd.NewOptimizer(updater, nn.NewDefaultParamsIterator(t.model),
		gd.ClipGradByValue(GradientClipTreshold))

	for epoch := 0; epoch < trainingParams.NumEpochs; epoch++ {
		t.optimizer.IncEpoch()
		for i, batch := range data {
			totalLoss, classificationLoss, sparsityLoss := t.trainBatch(&batch)
			t.optimizer.Optimize()
			if i%t.params.ReportInterval == 0 {
				log.Printf("Epoch %d batch %d loss %.5f | %.5f | %.5f \n", epoch, i, totalLoss, classificationLoss, sparsityLoss)
			}
		}
	}

	model := model.Model{
		MetaData: metaData,
		TabNet:   t.model,
	}

	outputFile, err := os.Create(outputFileName)
	if err != nil {
		log.Printf("Error creating output file %s: %s", outputFileName, err)
	}
	defer outputFile.Close()

	err = io.SaveModel(&model, outputFile)
	if err != nil {
		log.Printf("Error saving model to %s: %s", outputFileName, err)
	}

}

func (t *Trainer) trainBatch(batch *io.DataBatch) (float64, float64, float64) {
	t.optimizer.IncBatch()

	g := ag.NewGraph(ag.Rand(rand.NewLockedRand(t.params.RndSeed)))
	defer g.Clear()
	input := make([]ag.Node, len(batch.ContinuousFeatures))
	//TODO: add support for categorical features
	for i := range input {
		input[i] = g.NewVariable(batch.ContinuousFeatures[i], false)
	}
	modelProc := t.model.NewProc(g).(*model.TabNetProcessor)
	logits := modelProc.Forward(input...)
	loss := g.NewVariable(mat.NewScalar(0), true)
	classificationLoss := g.NewVariable(mat.NewScalar(0), true)
	sparsityLoss := g.NewVariable(mat.NewScalar(0), true)
	for i := range batch.Targets {
		exampleCrossEntropy := losses.CrossEntropy(g, logits[i], int(batch.Targets[i]))
		classificationLoss = g.Add(classificationLoss, exampleCrossEntropy)
		sparsityLoss = g.Add(sparsityLoss, modelProc.AttentionEntropy[i])
		exampleAttentionEntropy := g.Mul(modelProc.AttentionEntropy[i], g.Constant(t.model.SparsityLossWeight))
		exampleLoss := g.Add(exampleCrossEntropy, exampleAttentionEntropy)
		loss = g.Add(loss, exampleLoss)
	}
	batchSize := g.NewScalar(float64(len(batch.Targets)))
	loss = g.Div(loss, batchSize)
	classificationLoss = g.Div(classificationLoss, batchSize)
	sparsityLoss = g.Div(sparsityLoss, batchSize)

	g.Backward(loss)
	return loss.ScalarValue(), classificationLoss.ScalarValue(), sparsityLoss.ScalarValue()
}
