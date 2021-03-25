package pkg

import (
	"fmt"
	gio "io"
	"math"

	"sort"

	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/rs/zerolog/log"
	"gonum.org/v1/gonum/stat"

	"golem/pkg/io"
	"golem/pkg/model"

	"os"

	rand "github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/stats"
)

type NoopWriter struct{}

func (x NoopWriter) Write(p []byte) (n int, err error) {
	return len(p), nil
}

func printDataErrors(errors []io.DataError) {
	for _, err := range errors {
		log.Error().Msgf("Error parsing data at line %d: %s\n", err.Line, err.Error)
	}
}

func Test(modelFileName, inputFileName, outputFileName string, attentionFileName string) error {

	modelFile, err := os.Open(modelFileName)
	if err != nil {
		return fmt.Errorf("error opening model file %s: %w", modelFileName, err)
	}

	model, err := io.LoadModel(modelFile)
	if err != nil {
		return fmt.Errorf("error loading model from file %s: %w", modelFileName, err)
	}
	_, dataSet, dataErrors, err := io.LoadData(io.DataParameters{
		DataFile:           inputFileName,
		TargetColumn:       model.MetaData.Columns[model.MetaData.TargetColumn].Name,
		CategoricalColumns: nil,
		BatchSize:          1,
	}, model.MetaData)
	if err != nil {
		return fmt.Errorf("error loading data from %s: %w", inputFileName, err)
	}
	printDataErrors(dataErrors)
	if len(dataSet.Data) == 0 {
		log.Fatal().Msg("No data to test")
		return nil
	}
	return testInternal(model, dataSet, outputFileName, attentionFileName)
}

type modelEvaluator interface {
	Columns() []string
	EvaluatePrediction(prediction ag.Node, record *io.DataRecord) []string
	LogMetrics()
	Loss() float64
}

type classificationEvaluator struct {
	predictionCount int
	loss            float64
	metrics         map[string]*stats.ClassMetrics
	model           *model.Model
	lossFunc        lossFunc
	g               *ag.Graph
}
type classificationPrediction struct {
	predictedClass string
	label          string
	labelValue     mat.Float
	logits         mat.Matrix
	maxLogit       mat.Float
}

func (c *classificationEvaluator) Columns() []string {
	return []string{"label", "predicted", "probability"}
}
func (c *classificationEvaluator) EvaluatePrediction(node ag.Node, record *io.DataRecord) []string {
	prediction := c.decode(node, record)
	c.loss += float64(c.lossFunc(c.g, c.g.NewVariable(prediction.logits, false), prediction.labelValue).ScalarValue())
	c.predictionCount++

	result := []string{prediction.label, prediction.predictedClass, fmt.Sprintf("%.5f", toProbability(prediction.maxLogit))}

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

	return result
}

func (c *classificationEvaluator) LogMetrics() {
	// Sort class names for deterministic output
	sortedClasses := sortClasses(c.metrics)
	for _, class := range sortedClasses {
		result := c.metrics[class]
		log.Info().Str("Class", class).
			Int("TP", result.TruePos).
			Int("FP", result.FalsePos).
			Int("TN", result.TrueNeg).
			Int("FN", result.FalseNeg).
			Float32("Precision", result.Precision()).
			Float32("Recall", result.Recall()).
			Float32("F1", result.F1Score()).
			Msg("")

	}

	microF1, macroF1 := computeOverallF1(c.metrics)
	log.Info().Float64("MacroF1", macroF1).Float64("MicroF1", microF1).Msg("")

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

func toProbability(logit mat.Float) float64 {
	v := math.Exp(float64(logit))
	return v / (1.0 + v)

}
func testInternal(m *model.Model, dataSet *io.DataSet, outputFileName, attentionFileName string) error {

	var predictionOutput gio.Writer
	var attentionOutput gio.Writer

	if outputFileName != "" {
		outputFile, err := os.Create(outputFileName)
		if err != nil {
			return fmt.Errorf("error opening output file %s: %w", outputFileName, err)
		}
		defer outputFile.Close()
		predictionOutput = outputFile
	} else {
		predictionOutput = NoopWriter{}
	}

	if attentionFileName != "" {
		attentionFile, err := os.Create(attentionFileName)
		if err != nil {
			return fmt.Errorf("error creating attention output file %s:%w", attentionFileName, err)
		}
		defer attentionFile.Close()
		attentionOutput = attentionFile
	} else {
		attentionOutput = NoopWriter{}
	}

	attnWriter := &attentionWriter{
		outputWriter: attentionOutput,
		metaData:     m.MetaData,
	}

	lossFunc := lossFor(m.MetaData)

	g := ag.NewGraph(ag.Rand(rand.NewLockedRand(42)),
		ag.ConcurrentComputations(1))

	var evaluator modelEvaluator
	switch m.MetaData.TargetType() {
	case model.Categorical:
		evaluator = &classificationEvaluator{
			metrics:  map[string]*stats.ClassMetrics{},
			model:    m,
			lossFunc: lossFunc,
			g:        g,
		}
	default:
		evaluator = &regressionEvaluator{
			lossFunc:     lossFunc,
			g:            g,
			targetColumn: m.MetaData.Columns[m.MetaData.TargetColumn],
		}
	}

	dataSet.ResetOrder(io.OriginalOrder)
	ctx := nn.Context{Graph: g, Mode: nn.Inference}
	proc := nn.Reify(ctx, m.TabNet).(*model.TabNet)
	recLoss := 0.0
	sparsityLoss := 0.0
	numPredictions := 0

	outputColumns := evaluator.Columns()
	outputColumns = append(outputColumns, "reconstructionLoss")
	for i := 0; i < len(outputColumns)-1; i++ {
		fmt.Fprintf(predictionOutput, "%s,", outputColumns[i])
	}
	fmt.Fprintf(predictionOutput, "%s\n", outputColumns[len(outputColumns)-1])

	for d := dataSet.Next(); len(d) > 0; d = dataSet.Next() {
		normalizedInput, output := predict(g, proc, d)
		for i, prediction := range output.Output {
			evalOutput := evaluator.EvaluatePrediction(prediction, d[i])
			attnWriter.writeStepAttentionMap(output.AttentionMasks[i])
			predReconstructionLoss := float64(reconstructionLoss(g, normalizedInput[i], output.DecoderOutput[i]).ScalarValue())

			for _, v := range evalOutput {
				fmt.Fprintf(predictionOutput, "%s,", v)
			}
			fmt.Fprintf(predictionOutput, "%f\n", predReconstructionLoss)

			recLoss = recLoss + predReconstructionLoss
			sparsityLoss = sparsityLoss + float64(output.AttentionEntropy[i].ScalarValue())
			numPredictions++
		}
		g.Clear()
	}
	evaluator.LogMetrics()
	recLoss = recLoss / float64(numPredictions)
	sparsityLoss = sparsityLoss / float64(numPredictions)
	log.Info().Float64("Loss", evaluator.Loss()).
		Float64("ReconstructionLoss", recLoss).
		Float64("SparsityLoss", sparsityLoss).Msg("")

	return nil
}

func computeOverallF1(metrics map[string]*stats.ClassMetrics) (float64, float64) {
	macroF1 := 0.0
	for _, metric := range metrics {
		macroF1 += float64(metric.F1Score())
	}
	macroF1 /= float64(len(metrics))

	micro := stats.NewMetricCounter()
	for _, result := range metrics {
		micro.TruePos += result.TruePos
		micro.FalsePos += result.FalsePos
		micro.FalseNeg += result.FalseNeg
		micro.TrueNeg += result.TrueNeg
	}
	return macroF1, float64(micro.F1Score())

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
	loss            mat.Float
	predictionCount int
	estimated       []mat.Float
	values          []mat.Float
	lossFunc        lossFunc
	g               *ag.Graph
	targetColumn    *model.Column
}

func (r *regressionEvaluator) Columns() []string {
	return []string{"label", "prediction"}
}
func (r *regressionEvaluator) originalTargetValue(v mat.Float) float64 {
	return float64(v)*r.targetColumn.StdDev + r.targetColumn.Average

}
func (r *regressionEvaluator) EvaluatePrediction(prediction ag.Node, record *io.DataRecord) []string {

	result := []string{fmt.Sprintf("%f", r.originalTargetValue(record.Target)), fmt.Sprintf("%f", r.originalTargetValue(prediction.ScalarValue()))}

	r.estimated = append(r.estimated, prediction.ScalarValue())
	r.values = append(r.values, record.Target)
	r.loss += r.lossFunc(r.g, prediction, record.Target).ScalarValue()
	r.predictionCount++

	return result
}

func (r *regressionEvaluator) LogMetrics() {
	estimated := make([]float64, len(r.estimated))
	values := make([]float64, len(r.values))
	for i := range r.estimated {
		estimated[i] = float64(r.estimated[i])
	}
	for i := range r.values {
		values[i] = float64(r.values[i])
	}
	r2 := stat.RSquaredFrom(estimated, values, nil)
	log.Info().Float64("R-squared", r2).Msg("")
}

func (r *regressionEvaluator) Loss() float64 {
	return float64(r.loss) / float64(r.predictionCount)
}

func predict(g *ag.Graph, m *model.TabNet, data io.DataBatch) ([]ag.Node, *model.TabNetOutput) {
	input := createInputNodes(data, g, m)
	return input, m.Forward(input)
}

type attentionWriter struct {
	outputWriter gio.Writer
	line         int
	wroteHeader  bool
	metaData     *model.Metadata
}

func (w *attentionWriter) writeStepAttentionMap(att model.AttentionMask) {
	w.writeHeader()
	for i := range att {
		fmt.Fprintf(w.outputWriter, "%d,%d,", w.line, i)
		for step := range att[i] {
			fmt.Fprintf(w.outputWriter, "%.3f", att[i][step])
			if step < len(att[i])-1 {
				fmt.Fprintf(w.outputWriter, ",")
			}
		}
		fmt.Fprintf(w.outputWriter, "\n")
	}
	w.line++

}

func (w *attentionWriter) writeHeader() {
	if w.wroteHeader {
		return
	}
	fmt.Fprintf(w.outputWriter, "line,step,")
	for i := range w.metaData.Columns {
		if i != w.metaData.TargetColumn {
			fmt.Fprintf(w.outputWriter, "%s", w.metaData.Columns[i].Name)
			if i < len(w.metaData.Columns)-2 {
				fmt.Fprintf(w.outputWriter, ",")
			}
		}
	}
	fmt.Fprintf(w.outputWriter, "\n")
	w.wroteHeader = true

}

func argmax(data []mat.Float) (int, mat.Float) {
	maxInd := 0
	for i := range data {
		if data[i] > data[maxInd] {
			maxInd = i
		}
	}
	return maxInd, data[maxInd]
}
