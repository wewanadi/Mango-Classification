# Mango-Classification
###### tags:`Image Classification` `EfficientNet` `Tensorflow` `Keras`

## Outcomes

## EfficientNet

## Data Pre-processing
### Data input
* Image stored in numpy.
  * image = (6400, ?, ?, 3) 
  * target = (6400, 1)

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
