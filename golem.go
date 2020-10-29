package main

import (
	"github.com/spf13/cobra"
	"golem/io"
)

func TrainCommand() *cobra.Command {

	var trainFile string
	var outputFile string

	var cmd = &cobra.Command{
		Use:   "Use: train -i trainData -o outputFile",
		Short: "Trains a new model on the provided training data and saves the trained model",
		Args:  cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return io.train(trainFile, outputFile)
		},
	}
	cmd.Flags().StringVarP(&trainFile, "train-file", "i", "", "name of train input file")
	cmd.Flags().StringVarP(&outputFile, "output-file", "o", "", "name of the file to save model to.")

	cmd.MarkFlagRequired("train-file")
	cmd.MarkFlagRequired("output-file")

	return cmd
}

func main() {
	Main := &cobra.Command{Use: "Use: golem"}
	Main.AddCommand(TrainCommand())

	if err := Main.Execute(); err != nil {
		panic(err)
	}
}
