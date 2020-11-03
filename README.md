# Mini-Object-Detection

**Github Reference:- https://github.com/Bengemon825/TF_Object_Detection2020
**Also Youtube tutorials from:- https://www.youtube.com/playlist?list=PLAs-3cqyNbIjGzf50LckxBndLCd1EgB0w 
and https://www.youtube.com/playlist?list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku


**Usage : 


python xml_to_csv.py 

python generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record --image_dir=images/

python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=images/

python train.py --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config --logtostderr

python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_pets.config --trained_checkpoint_prefix training/model.ckpt-3207 --output_directory new_graph
