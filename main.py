import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
import prettytensor as pt

# local imports
import plot
import tools
from loader import img_size, num_channels, num_classes
from prepare_dataset import maybe_download_and_extract


dataset = maybe_download_and_extract()
images_train = dataset['test_images']
labels_train = dataset['test_labels']
images_test = dataset['test_images']
labels_test = dataset['test_labels']
cls_test = dataset['test_cls']

images_valid = dataset['valid_images']
labels_valid = dataset['valid_labels']
cls_valid = dataset['valid_cls']

with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32,
                       shape=[None, img_size, img_size, num_channels],
                       name='x')
    y_true = tf.placeholder(tf.float32,
                            shape=[None, num_classes],
                            name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

distorted_images = tools.pre_process(images=x, training=True,
                                     num_channels=num_channels,
                                     img_size_cropped=24)


def main_network(images, training):
    # Wrap the input images as a Pretty Tensor object.
    seq = pt.wrap(images).sequential()

    with pt.defaults_scope(activation_fn=tf.nn.relu):
        with seq.subdivide(2) as inception_1:
            inception_1[0].conv2d(kernel=1, depth=32, batch_normalize=True).conv2d(kernel=5, depth=64)
            inception_1[1].conv2d(kernel=1, depth=64).conv2d(kernel=3, depth=128)

        with seq.subdivide(2) as inception_2:
            inception_2[0].conv2d(kernel=3, depth=32).max_pool(kernel=2, stride=2)
            inception_2[1].conv2d(kernel=5, depth=64).max_pool(kernel=2, stride=2)

        y_pred, loss = seq.flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=128, name='layer_fc2').\
            softmax_classifier(num_classes=num_classes, labels=y_true)

        return y_pred, loss


def create_network(training):
    # Wrap the neural network in the scope named 'network'.
    # Create new variables during training, and re-use during testing.
    network_name = 'network' if not training else 'network_train'
    # network_name = 'network'
    with tf.variable_scope(network_name, reuse=not training):
        # Just rename the input placeholder variable for convenience.
        images = x

        # Create TensorFlow graph for pre-processing.
        images = pre_process(images=images, training=training)

        # Create TensorFlow graph for the main processing.
        y_pred, loss = main_network(images=images, training=training)

    return y_pred, loss


global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)
y_pred, loss = create_network(training=True)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

# y_pred, _ = create_network(training=False)
y_pred_cls = tf.argmax(y_pred, dimension=1)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

session = tf.Session()

save_dir = 'logs/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'model.ckpt')

try:
    print("Trying to restore last checkpoint ...")

    # Use TensorFlow to find the latest checkpoint - if any.
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

    # Try and load the data in the checkpoint.
    saver.restore(session, save_path=last_chk_path)

    # If we get to this point, the checkpoint was successfully loaded.
    print("Restored checkpoint from:", last_chk_path)
except:
    # If the above failed for some reason, simply
    # initialize all the variables for the TensorFlow graph.
    print("Failed to restore checkpoint. Initializing variables instead.")
    session.run(tf.global_variables_initializer())

train_batch_size = 64


def random_batch():
    # Number of images in the training-set.
    num_images = len(images_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch


def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch()

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # We also want to retrieve the global_step counter.
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # Print status to screen every 100 iterations (and last).
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

        # Save a checkpoint to disk every 1000 iterations (and last).
        if (i_global % 1000 == 0) or (i == num_iterations - 1):
            # Save all variables of the TensorFlow graph to a
            # checkpoint. Append the global_step counter
            # to the filename so we save the last several checkpoints.
            saver.save(session,
                       save_path=save_path,
                       global_step=global_step)

            print("Saved checkpoint.")
            print_valid_accuracy()

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256


def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred


def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.

    # Return the classification accuracy
    # and the number of correct classifications.
    return correct.mean(), correct.sum()


def print_valid_accuracy():
    correct, cls_pred = predict_cls(images=images_valid,
                                    labels=labels_valid,
                                    cls_true=cls_valid)

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)

    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Validation-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls(images=images_test,
                                    labels=labels_test,
                                    cls_true=cls_test)

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)

    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot.plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot.plot_confusion_matrix(cls_pred=cls_pred)


# TODO(bufnal): optimize function
optimize(num_iterations=200)

print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)


# Set the rounding options for numpy.
np.set_printoptions(precision=3, suppress=True)

# This has been commented out in case you want to modify and experiment
# with the Notebook without having to restart it.
log_dir = 'logs'
log_dir_fullpath = os.path.join(os.getcwd(), log_dir)
file_writer = tf.summary.FileWriter(log_dir_fullpath, session.graph)
session.close()
