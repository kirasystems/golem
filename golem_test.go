package main

import (
	"bytes"
	"io/ioutil"
	"log"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGolem(t *testing.T) {

	tests := []struct {
		Name                string
		TrainCmdLine        string
		TestCmdLine         string
		ExpectedTrainOutput []string
		ExpectedTestOutput  []string
	}{
		{
			Name:                "Iris",
			TrainCmdLine:        "train -i datasets/iris/iris.train -o $MODEL -t species --categorical-columns species -n 20 -s 3 --sparsity-loss-weight 0.01",
			TestCmdLine:         "test -m $MODEL -i datasets/iris/iris.test",
			ExpectedTrainOutput: []string{"Epoch 19"},
			ExpectedTestOutput:  []string{"Macro F1: 0.9"},
		},
		{
			Name:                "Breast Cancer",
			TrainCmdLine:        "train -t Class -i datasets/breast_cancer/breast-cancer.train -o $MODEL --categorical-columns Class,Age,Menopause,Tumor-size,Inv-nodes,Node-caps,Breast,Breast-quad,Irradiat  -s 3 -n 50",
			TestCmdLine:         "test -i datasets/breast_cancer/breast-cancer.test -m $MODEL ",
			ExpectedTrainOutput: []string{"Epoch 49"},
			ExpectedTestOutput:  []string{"Macro F1: 0.7"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {

			modelFile, err := ioutil.TempFile("", "")
			require.NoError(t, err)
			modelFileName := modelFile.Name()
			modelFile.Close()
			defer os.Remove(modelFileName)

			trainCmd := TrainCommand()
			trainCmd.SetArgs(createArgs(tt.TrainCmdLine, modelFileName))

			b := bytes.NewBufferString("")
			log.SetOutput(b)
			err = trainCmd.Execute()
			require.NoError(t, err)
			outBytes, err := ioutil.ReadAll(b)
			require.NoError(t, err)
			out := string(outBytes)
			require.False(t, strings.Contains(strings.ToLower(out), "error"))
			for _, expected := range tt.ExpectedTrainOutput {
				require.True(t, strings.Contains(out, expected))
			}

			testCmd := TestCommand()
			testCmd.SetArgs(createArgs(tt.TestCmdLine, modelFileName))
			b.Reset()
			err = testCmd.Execute()
			require.NoError(t, err)
			outBytes, err = ioutil.ReadAll(b)
			require.NoError(t, err)
			out = string(outBytes)
			for _, expected := range tt.ExpectedTestOutput {
				require.True(t, strings.Contains(out, expected), "Expected: %s , got: %s", expected, out)
			}

		})
	}

}

func createArgs(line, modelFileName string) []string {
	line = strings.Replace(line, "$MODEL", modelFileName, -1)
	result := strings.Split(line, " ")
	return result
}
