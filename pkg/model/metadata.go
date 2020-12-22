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

func (f *NameMap) ValueFor(name string) (int, bool) {
	value, ok := f.NameToIndex[name]
	if !ok {
		value = len(f.NameToIndex)
		f.NameToIndex[name] = value
		f.IndexToName[value] = name
	}
	return value, !ok
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

type CategoricalValue struct {
	// Column identifies the column corresponding to this value in the dataset
	Column int
	// Value is the plain value of this categorical value
	Value string
}

type CategoricalValuesMap struct {
	ValueToIndex map[CategoricalValue]int
	IndexToValue map[int]CategoricalValue
}

func (c *CategoricalValuesMap) ValueFor(value CategoricalValue) int {
	result, ok := c.ValueToIndex[value]
	if !ok {
		result = len(c.ValueToIndex)
		c.ValueToIndex[value] = result
		c.IndexToValue[result] = value
	}
	return result

}

func (c *CategoricalValuesMap) Size() int {
	return len(c.ValueToIndex)
}

type ColumnType int

const (
	Continuous ColumnType = iota
	Categorical
)

type Column struct {
	Name string
	Type ColumnType
}

type Metadata struct {
	Columns []Column

	// ContinuousFeaturesMap maps a data row column index to a dense matrix column index
	ContinuousFeaturesMap *ColumnMap

	// CategoricalFeaturesMap maps a data row column index to the categorical features index
	CategoricalFeaturesMap *ColumnMap

	// CategoricalValuesMap maps a given categorical column to a map from values to indexes
	CategoricalValuesMap *CategoricalValuesMap

	// TargetColumn points to the column in the data row that contains the prediction target
	TargetColumn int

	// TargetMap contains a mapping of target category names to target category indexes
	TargetMap *NameMap
}

func NewMetadata() *Metadata {
	return &Metadata{
		Columns:                nil,
		ContinuousFeaturesMap:  NewColumnMap(),
		CategoricalFeaturesMap: NewColumnMap(),
		CategoricalValuesMap:   NewCategoricalValuesMap(),
		TargetMap:              NewNameMap(),
	}
}

func NewCategoricalValuesMap() *CategoricalValuesMap {
	return &CategoricalValuesMap{
		ValueToIndex: map[CategoricalValue]int{},
		IndexToValue: map[int]CategoricalValue{},
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
	target, _ := d.TargetMap.ValueFor(value)
	return float64(target)

}
