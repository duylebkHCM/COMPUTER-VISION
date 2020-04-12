from Duy.preprocessing import ImageToArrayPreprocessor
from Duy.preprocessing import SimplePreprocessor
from Duy.dataset import SimpleDatasetLoader
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from imutils import paths
import imutils
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
help="path to pre-trained model")
ap.add_argument("-b", "--batch-size", type=int, default=32,help="size of mini-batches passed to network")

args = vars(ap.parse_args())

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
"dog", "frog", "horse", "ship", "truck"]

print("[INFO] loading network architecture and weights...")

model = load_model(args["model"])

(testData, testLabels) = cifar10.load_data()[1]
testData = testData.astype('float')/255.0
np.random.seed(42)
idxs = np.random.randint(0, len(testData), size=(10,))

testData, testLabels = testData[idxs], testLabels[idxs]
testLabels = testLabels.flatten()

probs = model.predict(testData, batch_size = args['batch_size'])
predictions = probs.argmax(axis=1)

for (i, prediction) in enumerate(predictions):
    image = testData[i].astype('float32')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = imutils.resize(image, width=128, inter = cv2.INTER_CUBIC)

    print("[INFO] predicted :{}, actual: {} ".format(labelNames[prediction], labelNames[testLabels[i]]))

    cv2.imshow("Image", image)
    cv2.waitKey(0)
