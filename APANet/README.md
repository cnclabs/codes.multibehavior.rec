# APANet
## Code
This is the source code for paper: Multi-behavior Recommendation with Action Pattern-aware Networks

## Requirements
* Python3

## Usage
First, run the file `datasets/preprocess.py` to preprocess the data.
```
cd datasets
python preprocess.py --dataset sample
```
Than, train and evaluate the APANet model.
```
python main.py --dataset sample
```
## Citation