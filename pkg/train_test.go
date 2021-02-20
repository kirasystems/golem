package pkg

import (
	"testing"

	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/stretchr/testify/require"
)

type testRand struct {
	values []float32
	index  int
}

func (t *testRand) Float() float32 {
	v := t.values[t.index]
	t.index = (t.index + 1) % len(t.values)
	return v
}
func TestInputDropout(t *testing.T) {

	r := rand.NewLockedRand(42)
	g := ag.NewGraph(ag.Rand(r))
	tr := testRand{
		values: []float32{0.0, 0.09, 0.101, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0},
	}
	dropout := NewDropoutPreprocessor(0.1, &tr, 10, 5)
	data := make([]ag.Node, 5)
	for i := range data {
		data[i] = g.NewVariable(mat.NewInitVecDense(10, 100.0), false)
	}
	output := dropout.process(g, data)
	require.Equal(t, len(data), len(output))
	require.Equal(t, len(data), len(dropout.CurrentMasks))

	for i := range output {
		mask := dropout.CurrentMasks[i]
		require.Equal(t, mask.Rows(), data[i].Value().Rows())
		require.Equal(t, mask.Columns(), data[i].Value().Columns())
		require.Equal(t, mask.Data(), []float32{1, 1, 0, 0, 1, 1, 0, 0, 1, 1})
		require.Equal(t, output[i].Value().Data(), []float32{100.0, 100.0, 0.0, 0.0, 100.0, 100.0, 0.0, 0.0, 100.0, 100.0})
	}

}
