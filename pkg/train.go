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
	BatchSize                int
	CategoricalEmbeddingSize int
	NumEpochs                int
	LearningRate             float64
	ReportInterval           int
	RndSeed                  uint64
}

type Trainer struct {
	params    TrainingParameters
	optimizer *gd.GradientDescent
	model     *model.TabNet
}

func Train(trainFile, outputFileName, targetColumn string, config model.TabNetConfig, trainingParams TrainingParameters) {
	t := &Trainer{params: trainingParams}

	rndGen := rand.NewLockedRand(trainingParams.RndSeed)

	metaData, data, dataErrors, err := io.LoadData(io.DataParameters{
		DataFile:           trainFile,
		TargetColumn:       targetColumn,
		CategoricalColumns: io.Set{},
		BatchSize:          trainingParams.BatchSize}, nil)

	if err != nil {
		log.Fatalf("Error reading training data: %s", err)
		return
	}
	printDataErrors(dataErrors)
	if len(data) == 0 {
		log.Fatalf("No data to train")
	}

	//Overwrite number of feature columns as this is only known after parsing the dataset
	config.NumColumns = metaData.FeatureCount()
	t.model = model.NewTabNet(config)
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
			totalLoss, classificationLoss, sparsityLoss := t.trainBatch(batch)
			t.optimizer.Optimize()
			if i%t.params.ReportInterval == 0 {
				log.Printf("Epoch %d batch %d loss %.5f | %.5f | %.5f \n", epoch, i, totalLoss, classificationLoss, sparsityLoss)
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

}

func (t *Trainer) trainBatch(batch io.DataBatch) (float64, float64, float64) {
	t.optimizer.IncBatch()

	g := ag.NewGraph(ag.Rand(rand.NewLockedRand(t.params.RndSeed))) // TODO: we might use the same random generator among the batches until we run them concurrently
	defer g.Clear()
	input := make([]ag.Node, len(batch))
	//TODO: add support for categorical features
	for i := range input {
		input[i] = g.NewVariable(batch[i].ContinuousFeatures, false)
	}
	modelProc := t.model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).(*model.TabNetProcessor)
	logits := modelProc.Forward(input...)

	var loss, classificationLoss, sparsityLoss ag.Node
	for i := range batch {
		exampleCrossEntropy := losses.CrossEntropy(g, logits[i], int(batch[i].Target))
		classificationLoss = g.Add(classificationLoss, exampleCrossEntropy)
		sparsityLoss = g.Add(sparsityLoss, modelProc.AttentionEntropy[i])
		exampleAttentionEntropy := g.Mul(modelProc.AttentionEntropy[i], g.Constant(t.model.SparsityLossWeight))
		exampleLoss := g.Add(exampleCrossEntropy, exampleAttentionEntropy)
		loss = g.Add(loss, exampleLoss)
	}
	batchSize := g.NewScalar(float64(len(batch)))
	loss = g.Div(loss, batchSize)
	classificationLoss = g.Div(classificationLoss, batchSize)
	sparsityLoss = g.Div(sparsityLoss, batchSize)

	g.Backward(loss)
	return loss.ScalarValue(), classificationLoss.ScalarValue(), sparsityLoss.ScalarValue()
}
