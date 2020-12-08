package model

// NameMap implements a bidirectional mapping between a name and an index
type NameMap struct {
	NameToIndex map[string]int
	IndexToName map[int]string
}

func (f NameMap) Set(name string, index int) {
	f.NameToIndex[name] = index
	f.IndexToName[index] = name
}

func (f NameMap) Size() int {
	return len(f.IndexToName)
}

func (f NameMap) ContainsName(name string) (int, bool) {
	index, ok := f.NameToIndex[name]
	return index, ok

}
func NewFeatureIndexMap() NameMap {
	return NameMap{
		NameToIndex: map[string]int{},
		IndexToName: map[int]string{},
	}
}

// ColumnMap is a bidirectional mapping between a column index and a dense matrix index
type ColumnMap struct {
	ColumnToIndex map[int]int
	IndexToColumn map[int]int
}

func (f ColumnMap) Set(column int, index int) {
	f.ColumnToIndex[column] = index
	f.IndexToColumn[index] = column
}

func (f ColumnMap) Size() int {
	return len(f.ColumnToIndex)
}

func (f ColumnMap) GetColumn(column int) (int, bool) {
	index, ok := f.ColumnToIndex[column]
	return index, ok
}
func NewColumnMap() ColumnMap {
	return ColumnMap{
		ColumnToIndex: map[int]int{},
		IndexToColumn: map[int]int{},
	}
}

type Metadata struct {
	Columns []string

	// ContinuousFeaturesMap maps a data row column index to a dense matrix column index
	ContinuousFeaturesMap ColumnMap

	// CategoricalFeaturesMap maps a data row column index to the categorical features index
	CategoricalFeaturesMap ColumnMap

	// TargetColumn points to the column in the data row that contains the prediction target
	TargetColumn int

	// TargetMap contains a mapping of target category names to target category indexes
	TargetMap NameMap

	// CategoricalEmbeddingSize is the size of each categorical feature embedding
	CategoricalEmbeddingSize int
}

func NewMetadata() *Metadata {
	return &Metadata{
		Columns:                nil,
		ContinuousFeaturesMap:  NewColumnMap(),
		CategoricalFeaturesMap: NewColumnMap(),
		TargetMap:              NewFeatureIndexMap(),
	}
}

func (d *Metadata) FeatureCount() int {
	return d.CategoricalFeaturesMap.Size() + d.ContinuousFeaturesMap.Size()
}

func (d *Metadata) ParseCategoricalTarget(value string) float64 {
	target, ok := d.TargetMap.ContainsName(value)
	if !ok {
		target = d.TargetMap.Size()
		d.TargetMap.Set(value, target)
	}
	return float64(target)
}
