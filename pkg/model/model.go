package model

import "github.com/nlpodyssey/spago/pkg/ml/nn"

type CategoricalFeatureEmbedding map[string]*nn.Param

type Model struct {
	MetaData *Metadata
	TabNet   *TabNet
	// @jjviana why do we need a two level map? Isn't enough a simple map[string]*nn.Param ?
	CategoricalFeatureEmbeddings map[int]CategoricalFeatureEmbedding // TODO: check if spaGO supports this
}
