# CHAI Datasets
This codebase is meant to be a universal data loader for PyTorch.

## Supported Datasets
- IEMOCAP
- MSP-Improv
- MSP-Podcast 
- MuSE
- Generic (works for data with no labels)

## Supported Acoustic Features
- MFBs 
- raw audio
- pre-extracted wav2vec2

## Supported Textual Features
- raw text
- pre-extracted BERT 

## Supported labels
Only Activation and Valence are currently supported. The values are min-max scaled into the range of -1 to 1, and then binned to return the continuous value and a binned value. 

## Supported Data Splits
### IEMOCAP
- Speaker Independent - {train: Session1-3, validation: Session4, test: Session5}
- No Lexical Repeat (Speaker Dependent) - impro/script sessions not repeated in different data splits. 
- Speaker Split - Returns a data split for each speaker in the dataset. 

### MSP-Improv 
- Speaker Independent
- No Lexical Repeat (Speaker Dependent) - impro/script sessions not repeated in different data splits. 
- Speaker Split - Returns a data split for each speaker in the dataset. 

### MSP-Podcast
- Data splits defined by MSP-Podcast (train/val/test_set_1/test_set_2)

### MuSE 
- Non stressed / Stressed data splits

## Additional Optional Features
- Segmentation (todo)

## Installation
Git clone the repository into your project and add any missing items from requirements.txt to your own project's requirements.txt and you should be good to go. 