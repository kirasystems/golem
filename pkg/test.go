package pkg

import (
	"fmt"

	gio "io"
	"log"
	"sort"

	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/losses"

	"golem/pkg/io"
	"golem/pkg/model"

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
	_, data, dataErrors, err := io.LoadData(io.DataParameters{
		DataFile:           inputFileName,
		TargetColumn:       model.MetaData.Columns[model.MetaData.TargetColumn],
		CategoricalColumns: nil,
		BatchSize:          1,
	}, model.MetaData)
	if err != nil {
		return fmt.Errorf("error loading data from %s: %w", inputFileName, err)
	}
	printDataErrors(dataErrors)
	if len(data) == 0 {
		log.Fatalf("No data to test")
	}
	return testInternal(model, data, outputFileName)
}
func testInternal(model *model.Model, data []io.DataBatch, outputFileName string) error {

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

	loss := 0.0
	numLosses := 0

	for _, d := range data {
		predictions := predict(g, model, d)
		for _, prediction := range predictions {

			loss += losses.CrossEntropy(g, g.NewVariable(prediction.logits, false), int(prediction.labelValue)).ScalarValue()
			numLosses++

			fmt.Fprintf(outputWriter, "%s,%s,%.5f\n", prediction.label, prediction.predictedClass, prediction.maxLogit)

			labelClassMetrics, ok := metrics[prediction.label]
			if !ok {
				labelClassMetrics = stats.NewMetricCounter()
				metrics[prediction.label] = labelClassMetrics
			}
			predictedClassMetrics, ok := metrics[prediction.predictedClass]
			if !ok {
				predictedClassMetrics = stats.NewMetricCounter()
				metrics[prediction.predictedClass] = predictedClassMetrics
			}

			if prediction.label == prediction.predictedClass {
				labelClassMetrics.IncTruePos()
			} else {
				labelClassMetrics.IncFalseNeg()
				predictedClassMetrics.IncFalsePos()
			}

		}

	}
	loss = loss / float64(numLosses)

	// Sort class names for deterministic output
	sortedClasses := sortClasses(metrics)
	for _, class := range sortedClasses {
		result := metrics[class]
		log.Printf("Class %s: TP %d FP %d TN %d FN %d Precision %.3f Recall %.3f F1 %.3f\n",
			class, result.TruePos, result.FalsePos, result.TrueNeg, result.FalseNeg, result.Precision(), result.Recall(),
			result.F1Score())
	}

	microF1, macroF1 := computeOverallF1(metrics)
	log.Printf("Macro F1: %.3f - Micro F1: %.3f - Loss %.5f\n", macroF1, microF1, loss)
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
	labelValue     float64
	logits         mat.Matrix
	maxLogit       float64
}

func predict(g *ag.Graph, model *model.Model, data io.DataBatch) []prediction {

	//TODO: add support for continuous outputs
	result := make([]prediction, data.Size())

	input := createInputNodes(data, g, model.TabNet)

	proc := model.TabNet.NewProc(nn.Context{Graph: g, Mode: nn.Inference})
	logits := proc.Forward(input...)
	for i := range logits {
		class, logit := argmax(logits[i].Value().Data())
		className := model.MetaData.TargetMap.IndexToName[class]
		label := model.MetaData.TargetMap.IndexToName[int(data[i].Target)]
		result[i] = prediction{
			predictedClass: className,
			label:          label,
			labelValue:     data[i].Target,
			logits:         logits[i].Value(),
			maxLogit:       logit,
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
