# Object detection

## Commands
```
# From tensorflow/models/
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# From tensorflow/models/object_detection/
# Prepare data and train
python3 xml_to_csv.py
python3 generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record
python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config

# Export graph from trained data
python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix training/model.ckpt-xxxx \
    --output_directory x_graph
```
