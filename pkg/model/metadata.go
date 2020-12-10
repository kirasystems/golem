package model

// NameMap implements a bidirectional mapping between a name and an index
type NameMap struct {
	NameToIndex map[string]int
	IndexToName map[int]string
}

func (f *NameMap) Set(name string, index int) {
	f.NameToIndex[name] = index
	f.IndexToName[index] = name
}

func (f *NameMap) ValueFor(name string) int {
	value, ok := f.NameToIndex[name]
	if !ok {
		value = len(f.NameToIndex)
		f.NameToIndex[name] = value
		f.IndexToName[value] = name
	}
	return value
}
func (f *NameMap) Size() int {
	return len(f.IndexToName)
}

func (f *NameMap) ContainsName(name string) (int, bool) {
	index, ok := f.NameToIndex[name]
	return index, ok

}
func NewNameMap() *NameMap {
	return &NameMap{
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
func NewColumnMap() *ColumnMap {
	return &ColumnMap{
		ColumnToIndex: map[int]int{},
		IndexToColumn: map[int]int{},
	}
}

type Metadata struct {
	Columns []string

	// ContinuousFeaturesMap maps a data row column index to a dense matrix column index
	ContinuousFeaturesMap *ColumnMap

	// CategoricalFeaturesMap maps a data row column index to the categorical features index
	CategoricalFeaturesMap *ColumnMap

	// CategoricalFeaturesValuesMap maps a given categorical column to a map from values to indexes
	CategoricalFeaturesValuesMap map[int]*NameMap

	// TargetColumn points to the column in the data row that contains the prediction target
	TargetColumn int

	// TargetMap contains a mapping of target category names to target category indexes
	TargetMap *NameMap

	// CategoricalEmbeddingSize is the size of each categorical feature embedding
	CategoricalEmbeddingSize int
}

func NewMetadata() *Metadata {
	return &Metadata{
		Columns:                      nil,
		ContinuousFeaturesMap:        NewColumnMap(),
		CategoricalFeaturesMap:       NewColumnMap(),
		CategoricalFeaturesValuesMap: map[int]*NameMap{},
		TargetMap:                    NewNameMap(),
	}
}

func (d *Metadata) FeatureCount() int {
	return d.CategoricalFeaturesMap.Size() + d.ContinuousFeaturesMap.Size()
}

func (d *Metadata) ParseCategoricalTarget(value string) (float64, bool) {
	index, ok := d.TargetMap.ContainsName(value)
	return float64(index), ok
}
func (d *Metadata) ParseOrAddCategoricalTarget(value string) float64 {
	return float64(d.TargetMap.ValueFor(value))

}
