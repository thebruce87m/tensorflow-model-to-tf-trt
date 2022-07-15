# tensorflow-model-to-tf-trt


# Run the docker
```bash
docker run \
--user $(id -u):$(id -g) \
-v /etc/passwd:/etc/passwd:ro \
-v /etc/group:/etc/group:ro \
--gpus all \
-it \
--rm \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-v $(pwd):/code \
-w /code/ \
nvcr.io/nvidia/tensorflow:22.06-tf2-py3
```

# Download a model

Models from here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

```bash
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_kpts_coco17_tpu-8.tar.gz

tar xzvf centernet_resnet50_v2_512x512_kpts_coco17_tpu-8.tar.gz
```

# Convert and save

Conversion code from here: https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#usage-example

```Python
from tensorflow.python.compiler.tensorrt import trt_convert as trt

input_saved_model_dir="/code/centernet_resnet50_v2_512x512_kpts_coco17_tpu-8/saved_model"

output_saved_model_dir="/code/output_model"


converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir)
converter.convert()
converter.save(output_saved_model_dir)
```
