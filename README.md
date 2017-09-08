# tensorflow-scripts
Custom scripts for machine learning

## Useful datasets

### [101 food categories](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)

## Dataset sources

### [Wikipedia - List of datasets for machine learning research](https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research)


### [UCI - Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.html)


### [GitHub - awesome-public-datasets](https://github.com/caesar0301/awesome-public-datasets)


### [Kaggle](https://www.kaggle.com/datasets)


### [Reddit - /r/datasets](https://www.reddit.com/r/datasets/)

## Useful software

### [labelImg](https://github.com/tzutalin/labelImg)

## Commands
```
# From tensorflow/models/
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# From tensorflow/models/object_detection/
python3 xml_to_csv.py
python3 generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record
python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config

python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix training/model.ckpt-xxxx \
    --output_directory x_graph
```
