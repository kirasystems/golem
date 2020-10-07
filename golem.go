package main

import (
	"golem/model"

	"github.com/spf13/cobra"
)

func TrainCommand() *cobra.Command {

	var trainFile string
	var outputFile string
	var targetColumn string
	var trainingParameters TrainingParameters
	var modelParameters model.TabNetParameters

	var cmd = &cobra.Command{
		Use:   "train -i trainData -o outputFile",
		Short: "Trains a new model on the provided training data and saves the trained model",
		Args:  cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			Train(trainFile, outputFile, targetColumn, modelParameters, trainingParameters)
			return nil
		},
	}

	cmd.Flags().StringVarP(&trainFile, "train-file", "i", "", "name of train input file")
	cmd.Flags().StringVarP(&outputFile, "output-file", "o", "", "name of the file to save model to.")
	cmd.Flags().IntVarP(&trainingParameters.BatchSize, "batch-size", "b", 16, "batch size")
	cmd.Flags().Float64VarP(&trainingParameters.LearningRate, "learning-rate", "l", 0.01, "learning rate")
	cmd.Flags().IntVarP(&trainingParameters.ReportInterval, "report-interval", "r", 10, "loss report interval")
	cmd.Flags().IntVarP(&trainingParameters.NumEpochs, "num-epochs", "n", 10, "number of epochs to train")
	cmd.Flags().Uint64VarP(&trainingParameters.RndSeed, "random-seed", "x", 42, "random seed")
	cmd.Flags().IntVarP(&modelParameters.NumDecisionSteps, "num-decision-steps", "s", 2, "number of decision steps")
	cmd.Flags().IntVarP(&modelParameters.FeatureDimension, "feature-dimension", "f", 4, "feature dimension")
	cmd.Flags().IntVarP(&modelParameters.OutputDimension, "output-dimension", "k", 4, "output dimension")
	cmd.Flags().Float64VarP(&modelParameters.RelaxationFactor, "relaxation-factor", "g", 1.5, "relaxation factor")
	cmd.Flags().Float64VarP(&modelParameters.BatchMomentum, "batch-momentum", "", 0.9, "batch momentum")
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

	var cmd = &cobra.Command{
		Use:   "test -m modelFile -i trainFile [-o outputFile]",
		Short: "Runs the provided model on the specified data input and stores the results in the output file if provided, or stdout",
		Args:  cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return Test(modelFile, inputFile, outputFile)
		},
	}

	cmd.Flags().StringVarP(&modelFile, "model", "m", "", "name of model to test")
	cmd.Flags().StringVarP(&inputFile, "input", "i", "", "name of data input file (optional, uses stdin if not present)")
	cmd.Flags().StringVarP(&outputFile, "output", "o", "", "name of output file (optional, uses stdout if not present)")
	_ = cmd.MarkFlagRequired("model")

	return cmd

}

func main() {
	Main := &cobra.Command{Use: "golem"}
	Main.AddCommand(TrainCommand())
	Main.AddCommand(TestCommand())

	if err := Main.Execute(); err != nil {
		panic(err)
	}
}
