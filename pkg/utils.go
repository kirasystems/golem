package pkg

import (
	"golem/pkg/io"
	"log"
)

func printDataErrors(errors []io.DataError) {
	for _, err := range errors {
		log.Printf("Error parsing data at line %d: %s\n", err.Line, err.Error)
	}
}
