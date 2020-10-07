package model

type Metadata struct {
	Columns []string

	// ContinuousFeaturesMap maps a data row column index to a dense matrix column index
	ContinuousFeaturesMap map[int]int

	// CategoricalFeaturesMap maps a data row column index to the categorical features index
	CategoricalFeaturesMap map[int]int

	// TargetColumn points to the column in the data row that contains the prediction target
	TargetColumn int

	// TargetMap contains a mapping of target category names to target category indexes
	TargetMap map[string]int

	// InverseTargetMap contains a mapping from target category indexes to target names
	InverseTargetMap map[int]string
}

func (d *Metadata) FeatureCount() int {
	return len(d.CategoricalFeaturesMap) + len(d.ContinuousFeaturesMap)
}

func (d *Metadata) ParseCategoricalTarget(value string) float64 {
	target, ok := d.TargetMap[value]
	if !ok {
		if d.TargetMap == nil {
			d.TargetMap = make(map[string]int)
			d.InverseTargetMap = make(map[int]string)
		}
		target = len(d.TargetMap)
		d.TargetMap[value] = target
		d.InverseTargetMap[target] = value
	}
	return float64(target)
}
