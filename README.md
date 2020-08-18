# Mango-Classification
###### tags:`Image Classification` `EfficientNet` `Tensorflow` `Keras`

## Outcomes

* Accuracy : 82.3125%

<img src=https://github.com/wewanadi/Mango-Classification/blob/master/.Image/scoreboard.png width="700">

## Fine tune Model

| Model | Accuracy on test data | Loss on test data |
| :-----: | :----: | :----: |
| [EfficientNet-B0](https://github.com/wewanadi/Mango-Classification/blob/master/Trained_Model/Mango_EfficientNetB0.h5) | 81.375% | 0.4152 | 
| [EfficientNet-B1](https://github.com/wewanadi/Mango-Classification/blob/master/Trained_Model/Mango_EfficientNetB1.h5) | 82.875% | 0.4052 | 
| EfficientNet-B2 | 83.25% | 0.4002 | 
| EfficientNet-B3 | 85.25% | 0.3800 | 
| EfficientNet-B4 | 84.5% | 0.3927 | 
| EfficientNet-B5 | 82.625% | 0.4189 | 

* Poor performance on bigger model because of batch-size getting smaller.(Train on 1 RTX-1080ti)
* B2, B3, B4, B5 is too large to upload on Github.

## EfficientNet

## Data Pre-processing
### Data input
* Image stored in numpy.
  * image = ```shape(6400, ?, ?, 3)```
  * target = ```shape(6400, 1)```

### Resize Image
* Make sure all Image are in the same shape.

### Setting up Data Generator
* Data generator in tensorflow.keras can help speed up the training.

## Model
### EfficientNet
* Getting EfficientNet by ```import efficientnet.tfkeras as efn``` with pretrain weights.

### Compile
* Optimizers: Adam(lr=0.0002 ,decay=1e-8)

### Call Back Functions
* ReduceLROnPlateau (monitor='val_loss', patience= 4, factor=0.5)
* EarlyStopping(monitor='val_loss', patience= 10, min')
* Save Best Model
```python
from tensorflow.keras.callbacks import Callback

class save_best_model(Callback):
    def on_epoch_end(self, epoch, logs=None):
        global best_loss
        global best_acc
        
        if best_loss > logs['val_loss']:  
            filepath='./Trained_model/{}Best_loss[{:.4f}].h5'.format(saving_name, best_loss)
            try:
                os.remove(filepath)
            except:
                None
            best_loss = logs['val_loss']
            filepath='./Trained_model/{}Best_loss[{:.4f}].h5'.format(saving_name, best_loss)
            model.save(filepath)
            print ('\nSave',filepath)
        else:
            print ('\nNot better loss than {:.4f}'.format(best_loss))
            
        if best_acc < logs['val_accuracy']:
            filepath='./Trained_model/{}Best_acc[{:.4f}].h5'.format(saving_name, best_acc*100)
            try:
                os.remove(filepath)
            except:
                None
            best_acc = logs['val_accuracy']
            filepath='./Trained_model/{}Best_acc[{:.4f}].h5'.format(saving_name, best_acc*100)
            model.save(filepath)
            print ('Save',filepath)
        else:
            print ('Not better acc than {:.4f}'.format(best_acc))
```
## Demo
* [Image to numpy](https://github.com/wewanadi/Mango-Classification/blob/master/Image2Numpy.ipynb)
* [Train](https://github.com/wewanadi/Mango-Classification/blob/master/Train.ipynb)
