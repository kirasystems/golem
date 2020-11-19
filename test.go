package main

import (
	"fmt"
	"golem/io"
	"golem/model"
	gio "io"
	"sort"

	"os"

	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/stats"
)

type NoopWriter struct{}

func (x NoopWriter) Write(p []byte) (n int, err error) {
	return len(p), nil
}

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
		outputFile, err := os.Create(outputFileName)
		if err != nil {
			return fmt.Errorf("error opening output file %s: %w", outputFileName, err)
		}
		defer outputFile.Close()
		outputWriter = outputFile
	} else {
		outputWriter = NoopWriter{}
	}

	metrics := make(map[string]*stats.ClassMetrics)

	g := ag.NewGraph(ag.Rand(rand.NewLockedRand(42)))

	for _, d := range data {
		predictions := predict(g, model, d)
		for _, prediction := range predictions {
			fmt.Fprintf(outputWriter, "%s,%s,%.5f\n", prediction.label, prediction.predictedClass, prediction.logit)

			labelClassMetrics, ok := metrics[prediction.label]
			if !ok {
				labelClassMetrics = stats.NewMetricCounter()
				metrics[prediction.label] = labelClassMetrics
			}
			predictedClassMetrics, ok := metrics[prediction.predictedClass]
			if !ok {
				predictedClassMetrics = stats.NewMetricCounter()
				metrics[prediction.predictedClass] = labelClassMetrics
			}

			if prediction.label == prediction.predictedClass {
				labelClassMetrics.IncTruePos()
			} else {
				labelClassMetrics.IncFalseNeg()
				predictedClassMetrics.IncFalsePos()
			}

		}

	}

	// Sort class names for deterministic output
	sortedClasses := sortClasses(metrics)
	for _, class := range sortedClasses {
		result := metrics[class]
		fmt.Printf("Class %s: TP %d FP %d TN %d FN %d Precision %.3f Recall %.3f F1 %.3f\n",
			class, result.TruePos, result.FalsePos, result.TrueNeg, result.FalseNeg, result.Precision(), result.Recall(),
			result.F1Score())
	}

	microF1, macroF1 := computeOverallF1(metrics)
	fmt.Printf("Macro F1: %.3f\nMicro F1: %.3f\n", macroF1, microF1)
	return nil
}

func computeOverallF1(metrics map[string]*stats.ClassMetrics) (float64, float64) {
	macroF1 := 0.0
	for _, metric := range metrics {
		macroF1 += metric.F1Score()
	}
	macroF1 /= float64(len(metrics))

	micro := stats.NewMetricCounter()
	for _, result := range metrics {
		micro.TruePos += result.TruePos
		micro.FalsePos += result.FalsePos
		micro.FalseNeg += result.FalseNeg
		micro.TrueNeg += result.TrueNeg
	}
	return macroF1, micro.F1Score()

}

func sortClasses(metrics map[string]*stats.ClassMetrics) []string {
	result := make([]string, 0, len(metrics))
	for class := range metrics {
		result = append(result, class)
	}
	sort.Strings(result)
	return result

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
