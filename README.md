# Golem

Golem is a batteries-included implementation of ["TabNet: Attentive Interpretable Tabular Learning"](https://arxiv.org/abs/1908.07442)

Some things set Golam apart from other TabNet implementations:

- It is written in Go, using the amazing [Spago](https://github.com/nlpodyssey/spago) module 
- More than just the core model, it provides a tool that can be used end-to-end, supporting the pervasive CSV file format out of the box.
- It implements simultaneously both the sparsity and input reconstruction objectives described in the original paper. Input reconstruction
loss can be useful for out-of-distribution detection. 
  
This is a work in progress that started as a platform for the exploration of the 
common territory between Software Engineering and Machine Learning. 

## Usage

### Data format

Golem takes CSV files as input. It expects CSV files to contain a column header in the first line.

### Train
`golem train -i <data file> -o <output file> -t <target column>`

Trains a model using the provided data. The data file should be in CSV format, and target column 
should be the name of the column to be predicted. 

The train command will create a model that tries to predict the variable in the target column using the other columns
as features. 

By default, all columns are considered to contain continuous variables. Columns representing
categorical variables should be named explicitly during training using the `--categorical-columns` option.

If the target column contains a continuous variable, Golem will build a regression model. Otherwise,
it will build a classification model.

There are options to control different aspects of training, like number of epochs, learning rate etc.
Please use `golem --help` for a complete list of options.

### Test
`golem test -i <data file> -m <model file> [-o output file]`

Loads the provided data file and model, uses the model to predict the target column, evaluates
the result and optionally writes each prediction to the output file.

The data file is expected to contain columns with the same name as in the training data file
for the model. It is not necessary to specify the nature (continuous or categorical) of each column,
since this information is saved during training.

## Credits

Thanks to [Matteo Grella](https://github.com/matteo-grella) for creating [Spago](https://github.com/nlpodyssey/spago)
and for many contributions to the code structure.

The name Golem is a homage to the book [Golem XIV](https://en.wikipedia.org/wiki/Golem_XIV)
by Stanislaw Lem.


