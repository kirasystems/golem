package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"testing"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"github.com/stretchr/testify/require"
)

type logLine map[string]interface{}

func parseOutputLog(out string) ([]logLine, error) {
	lines := strings.Split(out, "\n")
	result := make([]logLine, 0, len(lines))
	for _, line := range lines {
		if line == "" {
			continue
		}
		jsonLine := logLine{}
		err := json.Unmarshal([]byte(line), &jsonLine)
		if err != nil {
			return nil, err
		}
		result = append(result, jsonLine)

	}
	return result, nil
}
func hasExactValue(log []logLine, key string, value interface{}) bool {
	for _, line := range log {
		val := line[key]
		if val == value {
			return true
		}
	}
	return false
}
func getLastValue(log []logLine, key string) interface{} {
	for _, line := range log {
		val := line[key]
		if val != nil {
			return val
		}
	}
	return nil
}

type logExpectation struct {
	key                string
	exactValue         interface{}
	minValue, maxValue float64
}

func checkExpectation(log []logLine, expect logExpectation) error {
	if expect.exactValue != nil {
		if !hasExactValue(log, expect.key, expect.exactValue) {
			return fmt.Errorf("key %s not found in log with value %+v, log is: \n %+v", expect.key, expect.exactValue, log)
		}
		return nil
	}
	value := getLastValue(log, expect.key)
	if value == nil {
		return fmt.Errorf("key not found in log: %s", expect.key)
	}
	floatValue, ok := value.(float64)
	if !ok {
		return fmt.Errorf("value for key %s is not float: %+v", expect.key, value)
	}
	if floatValue < expect.minValue {
		return fmt.Errorf("value for %s (%f) is less than expected min %f", expect.key, floatValue, expect.minValue)
	}
	if floatValue > expect.maxValue {
		return fmt.Errorf("value for %s (%f) is larger than expected max %f", expect.key, floatValue, expect.maxValue)
	}
	return nil

}
func TestGolem(t *testing.T) {

	tests := []struct {
		Name                string
		TrainCmdLine        string
		TestCmdLine         string
		ExpectedTrainOutput []logExpectation
		ExpectedTestOutput  []logExpectation
	}{
		{
			Name:                "Iris",
			TrainCmdLine:        "train -i datasets/iris/iris.train -o $MODEL -t species --categorical-columns species -n 20 -s 3 --sparsity-loss-weight 0.01",
			TestCmdLine:         "test -m $MODEL -i datasets/iris/iris.test",
			ExpectedTrainOutput: []logExpectation{{key: "epoch", exactValue: 19.0}},
			ExpectedTestOutput:  []logExpectation{{key: "MacroF1", minValue: 0.85, maxValue: 1}},
		},
		{
			Name:                "Breast Cancer",
			TrainCmdLine:        "train -i datasets/breast_cancer/breast-cancer.train -o $MODEL -t Class --categorical-columns Class,Age,Menopause,Tumor-size,Inv-nodes,Node-caps,Breast,Breast-quad,Irradiat  -s 6 -n 40",
			TestCmdLine:         "test -i datasets/breast_cancer/breast-cancer.test -m $MODEL ",
			ExpectedTrainOutput: []logExpectation{{key: "epoch", exactValue: 39.0}},
			ExpectedTestOutput:  []logExpectation{{key: "MacroF1", minValue: 0.67, maxValue: 1}},
		},

		{
			Name:                "Boston Housing",
			TrainCmdLine:        "train -i datasets/boston_housing/boston-housing-train.csv -o $MODEL -t medv  -n 20 -s 3 --sparsity-loss-weight 0.01",
			TestCmdLine:         "test -i datasets/boston_housing/boston-housing-test.csv -m $MODEL ",
			ExpectedTrainOutput: []logExpectation{{key: "epoch", exactValue: 19.0}},
			ExpectedTestOutput:  []logExpectation{{key: "R-squared", minValue: 0.6, maxValue: 0.75}},
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
			log.Logger = zerolog.New(b)
			err = trainCmd.Execute()
			require.NoError(t, err)
			out, err := parseOutputLog(b.String())
			require.NoError(t, err)
			require.False(t, hasExactValue(out, "level", "fatal"))
			for _, expected := range tt.ExpectedTrainOutput {
				require.NoError(t, checkExpectation(out, expected))
			}

			testCmd := TestCommand()
			testCmd.SetArgs(createArgs(tt.TestCmdLine, modelFileName))
			b.Reset()
			err = testCmd.Execute()
			require.NoError(t, err)
			out, err = parseOutputLog(b.String())
			require.NoError(t, err)
			require.False(t, hasExactValue(out, "level", "fatal"))
			for _, expected := range tt.ExpectedTestOutput {
				require.NoError(t, checkExpectation(out, expected))
			}

		})
	}

}

func createArgs(line, modelFileName string) []string {
	line = strings.Replace(line, "$MODEL", modelFileName, -1)
	result := strings.Split(line, " ")
	return result
}
