package main

import (
	"bytes"
	"io/ioutil"
	"log"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestIris(t *testing.T) {

	trainCmd := TrainCommand()
	trainCmd.SetArgs(strings.Split("-i datasets/iris/iris.train -o /tmp/iris.model -t species -n 20 -s 3 --sparsity-loss-weight 0.01", " "))
	b := bytes.NewBufferString("")
	log.SetOutput(b)
	err := trainCmd.Execute()
	require.NoError(t, err)
	outBytes, err := ioutil.ReadAll(b)
	require.NoError(t, err)
	out := string(outBytes)
	require.False(t, strings.Contains(out, "Error"))
	require.True(t, strings.Contains(out, "Epoch 19"))
	require.False(t, strings.Contains(strings.ToLower(out), "error"))

	testCmd := TestCommand()
	testCmd.SetArgs(strings.Split("test -m /tmp/iris.model -i datasets/iris/iris.test", " "))
	b.Reset()
	err = testCmd.Execute()
	require.NoError(t, err)
	outBytes, err = ioutil.ReadAll(b)
	require.NoError(t, err)
	out = string(outBytes)
	require.True(t, strings.Contains(out, "Macro F1: 0.900"))
	require.False(t, strings.Contains(strings.ToLower(out), "error"))

}
