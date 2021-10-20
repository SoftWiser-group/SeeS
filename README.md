# SeeS
Python implementation of paper *Structure Meets Sequences: Predicting Network ofCo-evolving Sequences* (WSDM 2022)

## Requirements

- Python >= 3.6
- PyTorch >= 1.8

## Usage

For missing value recovery (single value prediction):

	$ python main.py

For future value prediction:

	$ python seesfuture.py
	
If you need to use other setting and dataset, please use the following command to see the detailed description:

	$ python main.py -h

## example


	$ python main.py --input motion/jump 
	...
	...
	Testing...
	Time: 0m 0s
	Testing loss: 0.737294614315033
	$ 

## Contents

TBD
	
## Datasets

For more datasets used in this paper, we provide ...
