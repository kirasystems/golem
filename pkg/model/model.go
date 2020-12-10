package model

import "github.com/nlpodyssey/spago/pkg/ml/nn"

type CategoricalFeatureEmbedding map[string]*nn.Param

type Model struct {
	MetaData *Metadata
	TabNet   *TabNet
}
