
'''Runs inference on a teacher model and saves the output to a specified TFRecord file'''
import tensorflow as tf
import numpy as np
from utils import facenet

import sys
import argparse
import math

#Helper functions

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    
def perform_inference(dataset, model_path, save_path, image_size = 160, batch_size = 128, mode = "dataset"):
    
    if mode == 'dataset':
        # Check that there are at least one training image per class
        for cls in dataset:
            assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')
        
        # Extract and show data stats
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
    elif mode == "paths":
        paths = dataset
        print('Number of images: %d' % len(paths))
    else:
        print("Error! Wrong mode specified")
        return
    
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=1)
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(model_path)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            print("Embedding size", embedding_size)
            
             # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            print("Number of epochs", nrof_batches_per_epoch)
            
            ## Intialize TFRecord writer
            # open the TFRecords file
            writer = tf.python_io.TFRecordWriter(save_path)
                
            for i in range(nrof_batches_per_epoch):
                print("\rRunning Epoch #{}".format(i), end='')
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size, False)
                # Whats a phase train placeholder??
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                for i in range(images.shape[0]):
                        image = images[i][:][:][:]
                        #Calculate the path of the image relative to its dataset
                        new_path = '{}/{}'.format(paths_batch[i].split('/')[-2], paths_batch[i].split('/')[-1])
                        feature = {
                        "image_name_path": _bytes_feature(tf.compat.as_bytes(new_path)),
                        "embedding": _float_feature(emb_array[start_index+i].tolist())
                            }
                        # Create an example protocol buffer
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
    
                        # Serialize to string and write on the file
                        writer.write(example.SerializeToString())
            writer.close()
            sys.stdout.flush()
            print("\nInference results have successfully been written to {}".format(save_path))

        
def main(args):
    print(args)
    print("fwrf")
    #Load in your dataset
    print("Loading dataset")
    dataset = facenet.get_dataset(args.dataset_path)
    perform_inference(dataset, args.model_path, args.save_path, args.img_size, args.batch_size)
    

print("running as script!")
parser = argparse.ArgumentParser()
parser.add_argument(
      'model_path',
      type=str,
      help='Path to teacher model'
  )
parser.add_argument(
      'dataset_path',
      type=str,
      help="Path to dataset"
  )
parser.add_argument(
      'save_path',
      type=str,
      help="Path where the TFRecords file will be stored"
  )
parser.add_argument(
    '--img_size',
    type=int,
    default=160,
    help='Dataset image size'
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=512,
    help='Batch size for running inference'
)
    
args = parser.parse_args()
main(args)

