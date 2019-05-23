# Satellite imagery-based prediction of advanced demographics using embedded space statistics

Pytorch Implementation of Satellite imagery-based prediction of advanced demographics using embedded space statistics.

### Data

This research utilized two types of data. One is the demographic information on target areas and the other is the corresponding satellite imagery. Both data types were collected from ArcGIS, which provides a publicly available data repository on maps and geographic information.

Demographic information is called the Esri Advanced Demographics is accessible by the ArcGIS GeoEnrichment Service API.
<li>Visit the ArcGis website <a href="https://doc.arcgis.com/en/Esri-demographics/data/global-intro.htm" rel="nofollow">for a comprehensive set of ready-to-use demographic layers</a></li>
<br>
Satellite images from the tiles of World Imagery
<li>Visit the ArcGis website <a href="https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9/" rel="nofollow">for the world and high-resolution satellite and aerial imagery</a></li>

<br>
In this repository, we uploaded **whole custom dataset** used for fine-tuning and **small sample data** randomly selected from original dataset.
You can find these data in ./data folder. (Custom dataset = "./data/proxy", Main dataset = "./data/sample_train", "./data/sample_test") 


### Required packages
The code has been tested running under Python 3.5.2. with the following packages installed (along with their dependencies):

- numpy == 1.15.4
- pandas == 0.23.4
- torch == 1.0.1.post2
- torchnet == 0.0.4
- torchvision == 0.2.2.post3
- scikit-learn == 0.19.1

<p>We recommend using the open data science platform <a href="https://www.continuum.io/downloads" rel="nofollow">Anaconda</a>.</p>


### Part 1. Data Pruning
* * *
##### How to Run
Default values of hyper-parameter are defined in parameters.py, data_pruning_parser().

```
usage: 1-data_pruning.py [-h] [--lr LR] [--batch-size BATCH_SIZE]
                         [--epochs EPOCHS]
                         [--checkpoint-epochs CHECKPOINT_EPOCHS]
                         [--evaluation-epochs EVALUATION_EPOCHS]
                         [--workers WORKERS] [--load] [--modelurl MODELURL]
                         [--train-path TRAIN_PATH] [--test-path TEST_PATH]

Data Pruning Parser

optional arguments:
  -h, --help            show this help message and exit
  --lr LR, --learning-rate LR
                        learning rate
  --batch-size BATCH_SIZE
                        batch size
  --epochs EPOCHS       total epochs
  --checkpoint-epochs CHECKPOINT_EPOCHS
                        checkpoint frequency
  --evaluation-epochs EVALUATION_EPOCHS
                        evaluation frequency
  --workers WORKERS     number of workers
  --load                load trained model
  --modelurl MODELURL   model path
  --train-path TRAIN_PATH
                        Train images directory path to remove uninhabited areas
  --test-path TEST_PATH
                        Test images directory path to remove uninhabited areas
```

##### Example
```                  
$ python3 1-data_pruning.py --train-path ./data/sample_train --test-path ./data/sample_test 
``` 





### Part 2. Extracting Embedding
* * *
##### How to Run
Default values of hyper-parameter are defined in parameters.py, extract_embeddings_parser().

```
usage: 2-extract_embeddings.py [-h] [--lr LR] [--batch-size BATCH_SIZE]
                               [--epochs EPOCHS]
                               [--checkpoint-epochs CHECKPOINT_EPOCHS]
                               [--evaluation-epochs EVALUATION_EPOCHS]
                               [--workers WORKERS] [--load]
                               [--modelurl MODELURL]

Extract Embeddings Parser

optional arguments:
  -h, --help            show this help message and exit
  --lr LR, --learning-rate LR
                        learning rate
  --batch-size BATCH_SIZE
                        batch size
  --epochs EPOCHS       total epochs
  --checkpoint-epochs CHECKPOINT_EPOCHS
                        checkpoint frequency
  --evaluation-epochs EVALUATION_EPOCHS
                        evaluation frequency
  --workers WORKERS     number of workers
  --load                load trained model
  --modelurl MODELURL   model path
```

##### Example
```
$ python3 2-extract_embeddings.py --batch-size 50 --epochs 100
```

##### Result
Extracted embeddings from satellite images are saved to "./data/sample_train/reduced", and "./data/sample_test/reduced"
The size of matrix would be differ as the number of satellite images is different from every districts. (# of satellite images X 512)



### Part 3. Predict Demographics
* * *
##### How to Run
Default values of hyper-parameter are defined in parameters.py, predict_demographics_parser().

```
usage: 3-predict_demographics.py [-h] [--idx IDX]

Predict Demographics Parser

optional arguments:
  -h, --help  show this help message and exit
  --idx IDX   select which demographics to predict, 0 to 51
```

##### Example
```
$ python3 3-predict_demographics.py --idx 0
```

##### Result
Prediction result (R-squared and Mean Squared Error) will be shown in command line. 
You can add some codes for saving results in any other file format.


