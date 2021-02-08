package io

import (
	"math/rand"
)

type DataSet struct {
	Data         []*DataRecord
	BatchSize    int
	Rand         *rand.Rand
	dataIndices  []int
	currentOrder []int
	currentIndex int
}

type DatasetOrder int

const (
	OriginalOrder DatasetOrder = iota
	RandomOrder
)

func (d *DataSet) ResetOrder(order DatasetOrder) {
	if d.currentOrder == nil {
		d.currentOrder = make([]int, len(d.dataIndices))
	}
	switch order {
	case OriginalOrder:
		copy(d.currentOrder, d.dataIndices)
	case RandomOrder:
		ind := d.Rand.Perm(len(d.currentOrder))
		for i := range ind {
			d.currentOrder[i] = d.dataIndices[ind[i]]
		}
	}

	d.currentIndex = 0
}
func (d *DataSet) Next() DataBatch {
	batch := make(DataBatch, 0, d.BatchSize)
	for ; d.currentIndex < len(d.currentOrder) && len(batch) < d.BatchSize; d.currentIndex++ {
		batch = append(batch, d.Data[d.currentOrder[d.currentIndex]])
	}
	return batch
}

func (d *DataSet) Size() int {
	return len(d.dataIndices)
}

func NewDataSet(data []*DataRecord, batchSize int) *DataSet {
	dataIndices := make([]int, len(data))
	for i := range dataIndices {
		dataIndices[i] = i
	}
	ds := &DataSet{Data: data, BatchSize: batchSize, dataIndices: dataIndices}
	ds.ResetOrder(OriginalOrder)
	return ds
}

func NewDataSetSplit(data []*DataRecord, batchSize int, indices []int) *DataSet {
	ds := &DataSet{
		Data: data, BatchSize: batchSize, dataIndices: indices}
	ds.ResetOrder(OriginalOrder)
	return ds
}

func (d *DataSet) RandomSplit(sizes ...int) []*DataSet {
	indices := make([]int, len(d.dataIndices))
	copy(indices, d.dataIndices)
	d.Rand.Shuffle(len(indices), func(i, j int) {
		tmp := indices[i]
		indices[i] = indices[j]
		indices[j] = tmp
	})
	splits := make([]*DataSet, len(sizes))
	idx := 0
	for i := range sizes {
		splitIndices := make([]int, sizes[i])
		for j := range splitIndices {
			splitIndices[j] = indices[idx]
			idx++
		}
		splits[i] = NewDataSetSplit(d.Data, d.BatchSize, splitIndices)
	}
	return splits

}
