package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"golem/pkg"
	"golem/pkg/model"

	"github.com/spf13/cobra"
)

func TrainCommand() *cobra.Command {

	var trainFile string
	var testFile string
	var outputFile string
	var targetColumn string
	var trainingParameters pkg.TrainingParameters
	var modelParameters model.TabNetConfig

	var cmd = &cobra.Command{
		Use:   "train -i trainData -o outputFile",
		Short: "Trains a new model on the provided training data and saves the trained model",
		Args:  cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			pkg.Train(trainFile, testFile, outputFile, targetColumn, modelParameters, trainingParameters)
			return nil
		},
	}

	cmd.Flags().StringVarP(&trainFile, "train-file", "i", "", "name of train file")
	cmd.Flags().StringVarP(&testFile, "test-file", "", "", "name of test file")
	cmd.Flags().StringVarP(&outputFile, "output-file", "o", "", "name of the file to save model to.")
	cmd.Flags().IntVarP(&trainingParameters.BatchSize, "batch-size", "b", 16, "batch size")
	cmd.Flags().Float64VarP(&trainingParameters.LearningRate, "learning-rate", "l", 0.01, "learning rate")
	cmd.Flags().IntVarP(&trainingParameters.ReportInterval, "report-interval", "r", 10, "loss report interval")
	cmd.Flags().IntVarP(&trainingParameters.NumEpochs, "num-epochs", "n", 10, "number of epochs to train")
	cmd.Flags().Uint64VarP(&trainingParameters.RndSeed, "random-seed", "x", 42, "random seed")
	cmd.Flags().StringSliceVarP(&trainingParameters.CategoricalColumns, "categorical-columns", "", nil, "list of columns holding categorical data")
	cmd.Flags().Float64VarP(&trainingParameters.InputDropout, "input-dropout-probability", "", 0.0, "probability of input dropout")

	cmd.Flags().IntVarP(&modelParameters.CategoricalEmbeddingDimension, "categorical-embedding-size", "c", 1, "size of categorical embeddings")
	cmd.Flags().IntVarP(&modelParameters.NumDecisionSteps, "num-decision-steps", "s", 2, "number of decision steps")
	cmd.Flags().IntVarP(&modelParameters.IntermediateFeatureDimension, "feature-dimension", "f", 4, "feature dimension")
	cmd.Flags().IntVarP(&modelParameters.OutputDimension, "output-dimension", "k", 4, "output dimension")
	cmd.Flags().Float64VarP(&modelParameters.RelaxationFactor, "relaxation-factor", "g", 1.5, "relaxation factor")
	cmd.Flags().Float64VarP(&modelParameters.BatchMomentum, "batch-momentum", "", 0.9, "batch momentum")
	cmd.Flags().Float64VarP(&modelParameters.SparsityLossWeight, "sparsity-loss-weight", "", 0.0001, "weight of the sparsity loss in total loss")
	cmd.Flags().Float64VarP(&modelParameters.ReconstructionLossWeight, "reconstruction-loss-weight", "", 0.0000, "weight of the reconstruction loss in total loss")
	cmd.Flags().Float64VarP(&modelParameters.TargetLossWeight, "target-loss-weight", "", 1.0000, "weight of the target loss in total loss")

	cmd.Flags().StringVarP(&targetColumn, "target-column", "t", "", "target column")

	_ = cmd.MarkFlagRequired("train-file")
	_ = cmd.MarkFlagRequired("output-file")
	_ = cmd.MarkFlagRequired("target-column")

	return cmd
}

func TestCommand() *cobra.Command {
	var modelFile string
	var inputFile string
	var outputFile string
	var attentionMapFile string

	var cmd = &cobra.Command{
		Use:   "test -m modelFile -i trainFile [-o outputFile] [-a attentionOutputFile]",
		Short: "Runs the provided model on the specified data input and optionally writes the results and attention map",
		Args:  cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return pkg.Test(modelFile, inputFile, outputFile, attentionMapFile)
		},
	}

	cmd.Flags().StringVarP(&modelFile, "model", "m", "", "name of model to test")
	cmd.Flags().StringVarP(&inputFile, "input", "i", "", "name of data input file (optional, uses stdin if not present)")
	cmd.Flags().StringVarP(&outputFile, "output", "o", "", "name of output file (optional)")
	cmd.Flags().StringVarP(&attentionMapFile, "attentionMap", "a", "", "name of attention map output file (optional)")

	_ = cmd.MarkFlagRequired("model")

	return cmd

}

var logLevel string
var logFormat string

func main() {

	Main := &cobra.Command{Use: "golem", PersistentPreRun: setupLogging}

	Main.PersistentFlags().StringVarP(&logLevel, "log-level", "", "info", "Logging level: info error or debug")
	Main.PersistentFlags().StringVarP(&logFormat, "log-format", "", "pretty", "Logging format: pretty or json")

	Main.AddCommand(TrainCommand())
	Main.AddCommand(TestCommand())

	if err := Main.Execute(); err != nil {
		panic(err)
	}
}

func setupLogging(cmd *cobra.Command, args []string) {

	switch logLevel {
	case "error":
		zerolog.SetGlobalLevel(zerolog.ErrorLevel)
	case "debug":
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	case "info":
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	default:
		panic("Invalid logging level specified")
	}

	switch logFormat {
	case "pretty":
		setupPrettyLogging()
	case "json":
	default:
		panic("Invalid log format specified")

	}

}

func setupPrettyLogging() {
	writer := zerolog.ConsoleWriter{Out: os.Stderr}
	writer.FormatFieldValue = func(i interface{}) string {
		switch v := i.(type) {
		case json.Number:
			val, _ := v.Float64()
			return fmt.Sprintf("%.3f", val)
		default:
			return fmt.Sprintf("%s", i)
		}

	}
	log.Logger = log.Output(writer)

}
