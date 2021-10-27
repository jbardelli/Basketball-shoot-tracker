""" Does the following steps for training TENSORFLOW MODEL
    1 - CREATES TF RECORDS
    2 - COPIES THE pipeline.config FILE
    3 - UPDATES THE config FILE
    """
import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from Paths import paths, files, labels
from Paths import PRETRAINED_MODEL_NAME

# Create TF Records for Training
print('Generating TF Records for Training')
os.system('python ' + files['TF_RECORD_SCRIPT']
          + ' -x ' + paths['IMAGE_PATH'] + '\\train'
          + ' -l ' + files['LABELMAP']
          + ' -o ' + paths['ANNOTATION_PATH'] + '\\train.record')

# Create TF Records for Testing
print('Generating TF Records for Testing')
os.system('python ' + files['TF_RECORD_SCRIPT']
          + ' -x ' + paths['IMAGE_PATH'] + '\\test'
          + ' -l ' + files['LABELMAP']
          + ' -o ' + paths['ANNOTATION_PATH'] + '\\test.record')

# Copy the pipeline.config from the pretrained model to my CUSTOM model
ORIGIN = os.path.join(paths['PRETRAINED_MODEL_PATH'],PRETRAINED_MODEL_NAME, 'pipeline.config')
DESTINATION = paths['CHECKPOINT_PATH']
print('Copying pipeline.config...')
os.system('copy ' + ORIGIN + ' ' + DESTINATION)

config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

STEPS = 200000
pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 15
pipeline_config.train_config.num_steps = STEPS
pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.total_steps = STEPS
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

config_text = text_format.MessageToString(pipeline_config)

with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
    f.write(config_text)