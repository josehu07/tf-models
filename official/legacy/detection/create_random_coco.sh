#!/bin/bash
NUM_RECORD=10000

rm -rf retinanet_data_gen
mkdir -p retinanet_data_gen/annotations
mkdir -p retinanet_data_gen/val2017

src_fnames=(000000397133 000000087038 000000458054 000000296649 000000386912 000000143931)

for (( i=1; i<=$NUM_RECORD; i++ )); do
	dst_fname=$(printf %012d $i).jpg
	src_idx=$[ $RANDOM % 6 ]
	src_fname=${src_fnames[$src_idx]}.jpg
	cp retinanet_data/train2017/$src_fname retinanet_data_gen/val2017/$dst_fname
done

python3 create_random_coco_annotation.py $NUM_RECORD
python3 create_coco_tfrecord/create_coco_tf_record.py --data_dir=./retinanet_data_gen --set=val --output_filepath=./retinanet_data_gen/val.tfrecord
