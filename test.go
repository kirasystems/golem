package main

import (
	"fmt"
	"golem/io"
	"golem/model"
	gio "io"

	"os"

	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

func Test(modelFileName, inputFileName, outputFileName string) error {

	modelFile, err := os.Open(modelFileName)
	if err != nil {
		return fmt.Errorf("error opening model file %s: %w", modelFileName, err)
	}

	model, err := io.LoadModel(modelFile)
	if err != nil {
		return fmt.Errorf("error loading model from file %s: %w", modelFileName, err)
	}

	_, data, err := io.LoadData(inputFileName, model.MetaData.Columns[model.MetaData.TargetColumn], 1)
	if err != nil {
		return fmt.Errorf("error loading data from %s: %w", inputFileName, err)
	}

	var outputWriter gio.Writer
	if outputFileName != "" {
		outputFile, err := os.Open(outputFileName)
		if err != nil {
			return fmt.Errorf("error opening output file %s: %w", outputFileName, err)
		}
		defer outputFile.Close()
		outputWriter = outputFile
	} else {
		outputWriter = os.Stdout
	}

	g := ag.NewGraph(ag.Rand(rand.NewLockedRand(42)))

	for _, d := range data {
		predictions := predict(g, model, d)
		for _, prediction := range predictions {
			fmt.Fprintf(outputWriter, "%s,%s,%.5f\n", prediction.label, prediction.predictedClass, prediction.logit)
		}

	}
	return nil
}

type prediction struct {
	predictedClass string
	label          string
	logit          float64
}

func predict(g *ag.Graph, model *model.Model, data io.DataBatch) []prediction {

	//TODO: add support cor continuous outputs
	result := make([]prediction, data.Size())

	input := make([]ag.Node, data.Size())
	//TODO: add support for categorical features
	for i := range data.ContinuousFeatures {
		input[i] = g.NewVariable(data.ContinuousFeatures[i], false)
	}

	proc := model.TabNet.NewProc(g)
	proc.SetMode(nn.Inference)
	logits := proc.Forward(input...)
	for i := range logits {
		class, logit := argmax(logits[i].Value().Data())
		className := model.MetaData.InverseTargetMap[class]
		label := model.MetaData.InverseTargetMap[int(data.Targets[i])]
		result[i] = prediction{
			predictedClass: className,
			label:          label,
			logit:          logit,
		}
	}
	g.Clear()
	return result

}

func argmax(data []float64) (int, float64) {
	maxInd := 0
	for i := range data {
		if data[i] > data[maxInd] {
			maxInd = i
		}
	}
	return maxInd, data[maxInd]
}
