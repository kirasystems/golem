package pkg

import (
	"fmt"
	gio "io"
	"log"
	"sort"

	"github.com/nlpodyssey/spago/pkg/mat"
	"gonum.org/v1/gonum/stat"

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
		TargetColumn:       model.MetaData.Columns[model.MetaData.TargetColumn].Name,
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

type modelEvaluator interface {
	EvaluatePrediction(prediction ag.Node, record *io.DataRecord)
	PrintMetrics(logger func(format string, v ...interface{}))
	Loss() float64
}

type classificationEvaluator struct {
	predictionCount int
	loss            float64
	metrics         map[string]*stats.ClassMetrics
	model           *model.Model
	lossFunc        lossFunc
	g               *ag.Graph
	outputWriter    gio.Writer
}
type classificationPrediction struct {
	predictedClass string
	label          string
	labelValue     float64
	logits         mat.Matrix
	maxLogit       float64
}

func (c *classificationEvaluator) EvaluatePrediction(node ag.Node, record *io.DataRecord) {
	prediction := c.decode(node, record)
	c.loss += c.lossFunc(c.g, c.g.NewVariable(prediction.logits, false), prediction.labelValue).ScalarValue()
	c.predictionCount++

	fmt.Fprintf(c.outputWriter, "%s,%s,%.5f\n", prediction.label, prediction.predictedClass, prediction.maxLogit)

	labelClassMetrics, ok := c.metrics[prediction.label]
	if !ok {
		labelClassMetrics = stats.NewMetricCounter()
		c.metrics[prediction.label] = labelClassMetrics
	}
	predictedClassMetrics, ok := c.metrics[prediction.predictedClass]
	if !ok {
		predictedClassMetrics = stats.NewMetricCounter()
		c.metrics[prediction.predictedClass] = predictedClassMetrics
	}

	if prediction.label == prediction.predictedClass {
		labelClassMetrics.IncTruePos()
	} else {
		labelClassMetrics.IncFalseNeg()
		predictedClassMetrics.IncFalsePos()
	}

}

func (c *classificationEvaluator) PrintMetrics(logger func(format string, v ...interface{})) {
	// Sort class names for deterministic output
	sortedClasses := sortClasses(c.metrics)
	for _, class := range sortedClasses {
		result := c.metrics[class]
		logger("Class %s: TP %d FP %d TN %d FN %d Precision %.3f Recall %.3f F1 %.3f\n",
			class, result.TruePos, result.FalsePos, result.TrueNeg, result.FalseNeg, result.Precision(), result.Recall(),
			result.F1Score())
	}

	microF1, macroF1 := computeOverallF1(c.metrics)
	logger("Macro F1: %.3f - Micro F1: %.3f", macroF1, microF1)
}

func (c *classificationEvaluator) Loss() float64 {
	return c.loss / float64(c.predictionCount)
}

func (c *classificationEvaluator) decode(modelOutput ag.Node, record *io.DataRecord) classificationPrediction {
	class, logit := argmax(modelOutput.Value().Data())
	className := c.model.MetaData.TargetMap.IndexToName[class]
	label := c.model.MetaData.TargetMap.IndexToName[int(record.Target)]
	return classificationPrediction{
		predictedClass: className,
		label:          label,
		labelValue:     record.Target,
		logits:         modelOutput.Value().Clone(),
		maxLogit:       logit,
	}
}
func testInternal(m *model.Model, data []io.DataBatch, outputFileName string) error {

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

	lossFunc := lossFor(m.MetaData)

	g := ag.NewGraph(ag.Rand(rand.NewLockedRand(42)))

	var evaluator modelEvaluator
	switch m.MetaData.TargetType() {
	case model.Categorical:
		evaluator = &classificationEvaluator{
			metrics:      map[string]*stats.ClassMetrics{},
			model:        m,
			lossFunc:     lossFunc,
			g:            g,
			outputWriter: outputWriter,
		}
	default:
		evaluator = &regressionEvaluator{
			lossFunc: lossFunc,
			g:        g,
		}
	}

	for _, d := range data {
		predictions := predict(g, m, d)
		for i, prediction := range predictions {
			evaluator.EvaluatePrediction(prediction, d[i])
		}
		g.Clear()

	}
	evaluator.PrintMetrics(log.Printf)
	log.Printf("Loss %.5f", evaluator.Loss())

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

type regressionEvaluator struct {
	loss            float64
	predictionCount int
	estimated       []float64
	values          []float64
	lossFunc        lossFunc
	g               *ag.Graph
}

func (r *regressionEvaluator) EvaluatePrediction(prediction ag.Node, record *io.DataRecord) {
	log.Printf("Test: target %.3f prediction %.3f", record.Target, prediction.ScalarValue())
	r.estimated = append(r.estimated, prediction.ScalarValue())
	r.values = append(r.values, record.Target)
	r.loss += r.lossFunc(r.g, prediction, record.Target).ScalarValue()
	r.predictionCount++
}

func (r *regressionEvaluator) PrintMetrics(logger func(format string, v ...interface{})) {
	r2 := stat.RSquaredFrom(r.estimated, r.values, nil)
	logger("R-squared: %.3f", r2)
}

func (r *regressionEvaluator) Loss() float64 {
	return r.loss / float64(r.predictionCount)
}

func predict(g *ag.Graph, m *model.Model, data io.DataBatch) []ag.Node {
	input := createInputNodes(data, g, m.TabNet)
	ctx := nn.Context{Graph: g, Mode: nn.Inference}
	proc := nn.Reify(ctx, m.TabNet).(*model.TabNet)
	result := proc.Forward(input)
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
