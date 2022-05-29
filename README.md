# Demo of HOG-SVM based mask detection
This is a mask detection application using Histogram of Oriented Gradients (HOG) as features and Support Vector Machines (SVM) as classifiers, using the Masked-Dataset dataset provided in the Pattern Recognition course for target detection training.

The program is programmed in Python and requires the following dependent libraries to run the program:
1. Scikit-learn (For implementing SVM)
2. Scikit-image (For HOG feature extraction)
3. OpenCV (for testing)
4. PIL (Image processing library)
5. Numpy (matrix multiplication)
6. Imutils for Non-maximum suppression

The training set should include the following.
1. positive sample images: these contain only the object you are trying to detect, e.g. a mouthpiece.
2. negative sample images: these images can contain anything except the object you are trying to detect.


The data set needs to be divided into the following two parts:
nag & pos, which correspond to the negative sample image storage folder and the positive sample image folder in the root directory of the program.

Train_HOG_SVM.py: for the training of the HOG_SVM model.

visualise_HOGdescriptors.py: visualises how the computed gradients will look on a given image (specified by the user).

testing_HOG_SVM.py: import an image and use the trained model_name.npy model file for the masks

Two screenshots of the program running during the actual test are given here. pos_result.png shows the detection with a mask (marked by a green box and the confidence score can be seen), while nag_result.png shows the detection without a mask.

```bash
Author：Huang Jiaqi
Created：2022-05-16
Last updated：2022-05-27
Function：Target detection task for masks using the HOG-SVM model for the Masked-Face-Dataset dataset.
```


