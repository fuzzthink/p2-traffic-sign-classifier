import glob, h5py, argparse
import np_util as npu 
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import __version__ as keras_version
from os.path import join


parser = argparse.ArgumentParser(description='Test Traffic Sign Classification Model')
parser.add_argument(
    'model',
    type=str,
    default='model.h5',
    help='Path to model h5 file. Model should be on the same path.'
)
parser.add_argument(
    'imagepath',
    type=str,
    default='5signs32x32',
    help='Path to images to test.'
)
args = parser.parse_args()

## check model compatibility
f = h5py.File(args.model, mode='r')
model_version = f.attrs.get('keras_version')
keras_version = str(keras_version).encode('utf8')
if model_version != keras_version:
    print('You are using Keras version ', keras_version,
          ', but the model was built using ', model_version)

model = load_model(args.model)


## Evaluate model on test data
testset = "data/test.p"
with open(testset, mode='rb') as f:
    test = pickle.load(f)
    
X_test, y_test = test['features'], test['labels']
X_test_gray = npu.grayscale(X_test)
X_test_norm = npu.normalize(X_test_gray)
y_test_hot = np_utils.to_categorical(y_test, n_classes)

metrics = model.evaluate(X_test_norm, y_test_hot)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('\n{}: {}'.format(metric_name, metric_value))


## Visualize classification on few test images
test2 = []
for i in glob.glob(join(args.imagepath, '*.jpg')):
    test2.append(cv2.imread(i))
    
test2 = np.asarray(test2)
Xtest2_gray = grayscale(test2)
Xtest2_norm = normalize(Xtest2_gray)

print("test2 shape after grayscale =", Xtest2_norm[0].shape)


probs = model.predict(Xtest2_norm)

sign_strs = ['No Pedestrian', 'Do Not Enter', 'No Exit', 'Road Work', 'Yield']
topk = 5

print('Sign          | Prediction')
for i in range(len(sign_strs)):
    clsmap = dict(zip(probs[i], range(n_classes)))
    sortedProb = sorted(probs[i], reverse=True)
    pct, cls, lbl, predictStrs = [],[],[],[]
    for k in range(topk):
        pct.append(sortedProb[k])
        cls.append(clsmap[pct[k]])
        lbl.append(labels[cls[k]])
        predictStr = '%05.2f%%:%s' % (pct[k]*100., lbl[k][:23])
        predictStrs.append('%-32s' % predictStr)
    print('%-13s | %s\n______________| %s\n' % (
        sign_strs[i], '| '.join(predictStrs[0:3]), '| '.join(predictStrs[3:5])))
    
j = 0
fig = plt.figure(figsize=(10,10))
for i in range(len(sign_strs)):
    j += 1
    plt.subplot(5,2,j)
    plt.title(sign_strs[i])
    plt.axis('off')
    plt.imshow(cv2.cvtColor(test2[i], cv2.COLOR_BGR2RGB))
    j += 1
    ax = fig.add_subplot(5,2,j)
    ax.bar(classIds, probs[i])
plt.subplots_adjust(top=1.2)
plt.imsave(args.imagepath.rstrip('/')+'.jpg') 

print(probs)