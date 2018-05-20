
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Convolution1D ,AveragePooling1D
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
np.random.seed(8888) 
sensor_num = 17
train = [open("/home/lab/cnn_ref_0325/data/sensor_cv/train_1/cv_1_train_%d.txt" % i) for i in range(1, sensor_num+1)]
test = [open("/home/lab/cnn_ref_0325/data/sensor_cv/test_1/cv_1_test_%d.txt" % i) for i in range(1, sensor_num+1)]


# In[2]:


WEIGHTS_FILEPATH = 'mc-dcnn_cv1_s1_relu.hdf5'
MODEL_ARCH_FILEPATH = 'mc-dcnn_cv1_s1_relu.json'


# In[3]:


def read_Data(txt):
    data = txt.readlines()
    data = [i.split('\n', 1)[0] for i in data]
    readlines = []
    for i in range(0,len(data)):
        readlines.append(data[i].split())
    return np.array(readlines)
train_sensors = []
test_sensors = []
for i in range(0,sensor_num):
    train_sensors.append(read_Data(train[i]))
    test_sensors.append(read_Data(test[i]))
    
#sensor_val[sensor_num][wafer_num]
train_length = np.array(train_sensors).shape[1]
test_length =  np.array(test_sensors).shape[1]
print("train dim=",np.array(train_sensors).shape)
print("test dim=",np.array(test_sensors).shape)


# In[4]:


def rstr(data):
    lenval = []
    print('sensor count = ',len(data))
    for i in range(0,len(data)):
        lenval.append(len(data[i]))
    print('MAX=',max(lenval))
    print('MIN=',min(lenval))
    return [np.array(lenval),min(lenval)]

def sensor_label(data):
    label = []
    for i in range(0,len(data[0])):
        label.append(data[0][i][0])
    return np.array(label)

train_label = sensor_label(train_sensors)
test_label = sensor_label(test_sensors)

print("train.y=",train_label,"length=",train_label.shape)
print("test.y=",test_label,"length=",test_label.shape)


# In[5]:


def sensor_value(data):
    for s in range(0,len(data)):
        for i in range(0,len(data[s])):
            data[s][i].pop(0)
    return data
                       
train_sensor_val = sensor_value(train_sensors)
test_sensor_val = sensor_value(test_sensors)

print(np.array(train_sensor_val[0][0][0:20]))
print(np.array(test_sensor_val[0][0][0:20]))


# In[6]:


np.array(test_sensors[0][0])[0:20] == np.array(test_sensor_val[0][0][0:20])


# In[7]:


rstr(train_sensor_val[0])


# In[8]:


rstr(test_sensor_val[0])


# In[9]:


min_len = min(rstr(train_sensor_val[0])[1],rstr(test_sensor_val[0])[1])


# In[10]:


def sliding_win(ts,w_size):
    window = []
    for i in range(0,len(ts)-w_size+1):
        window_v = ts[i:i+w_size]
        window.append(window_v)
    return window

def each_len(data,win_size):
    sliding_len = []
    for i in range(0,len(data[0])):
        sliding_len.append(len(sliding_win(data[0][i],win_size)))
    return np.array(sliding_len)

#slide each length
each_len(train_sensor_val,min_len)


# In[11]:


def wafer_idx(sensor,num_wafer,win_size):
    total = sum(each_len(sensor,win_size))
    val = each_len(sensor,win_size)
    idx = np.array(range(sum(val[0:num_wafer-1]),sum(val[0:num_wafer-1],val[num_wafer-1])))
    return idx

#each wafer after slide "index"!!! for predict label
wafer_idx(train_sensor_val,3,min_len-1)


# In[12]:


sensor_trainX = [sliding_win(train_sensor_val[0][i],min_len) for i in range(0,train_length)]
sensor_testX = [sliding_win(test_sensor_val[0][i],min_len) for i in range(0,test_length)]


# In[13]:


print("train=",len(sensor_trainX))
print("test=",len(sensor_testX))


# In[14]:


def unlist(df):
    unlist_val = []
    for i in range(0,len(df)):
        for s in range(0,len(df[i])):
            unlist_val.append(df[i][s])
    return unlist_val

def sensor_X(sensor,sensor_num,win_size):
    sensor_num=sensor_num-1
    sensor_dat = [sliding_win(sensor[sensor_num][i],win_size) for i in range(0, len(sensor[sensor_num]))]
    sensor_dat = np.asarray(unlist(sensor_dat))
    sensor_dat.reshape(sum(each_len(sensor,win_size)),win_size)
    return sensor_dat


# In[15]:


print("train:",np.shape(sensor_X(train_sensor_val,1,min_len)))
print("test:",np.shape(sensor_X(test_sensor_val,1,min_len)))


# In[16]:


min_len


# In[17]:


train_x = []
test_x = []
for i in range(0,17):
    train_x.append(sensor_X(train_sensor_val,i+1,min_len))
    test_x.append(sensor_X(test_sensor_val,i+1,min_len))
    #change type
    train_x[i] = train_x[i].astype(np.float32)
    test_x[i] = test_x[i].astype(np.float32)
    train_x[i] = train_x[i].reshape(np.shape(sensor_X(train_sensor_val,1,min_len))[0],min_len,1)
    test_x[i] = test_x[i].reshape(np.shape(sensor_X(test_sensor_val,1,min_len))[0],min_len,1)


# In[18]:


def y_label(data,label,win_size,wafer_num):
    n_label = []
    for i in wafer_num:
        n_label.append(np.repeat(label[i],each_len(data,win_size)[i]).astype('int'))
    return np.array(unlist(n_label))

train_y = y_label(train_sensor_val,train_label,149,range(0,len(train_sensor_val[0])))
test_y = y_label(test_sensor_val,test_label,149,range(0,len(test_sensor_val[0])))
print("test:",len(test_y))
print("train:",len(train_y))


# In[19]:


def process_output(labels):
    return to_categorical(labels, nb_classes=2)
train_y = process_output(train_y)
test_y = process_output(test_y)


# In[20]:


print('Building model...')
#input shape = (sliding window size,1)
def cnn_model(kernel_1,filter_1,act_fun):
    model = Sequential()
    model.add(Convolution1D(filter_1,kernel_1,input_shape=(149, 1)))
    model.add(Activation(act_fun))
    model.add(AveragePooling1D(pool_length=2))
    model.add(Flatten())
    return model


# In[21]:


model=[]
for i in range(0,17):
    model.append(cnn_model(5,8,'relu'))


# In[22]:


#merge model
merged = Merge(model, mode='concat')
final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(732, activation='relu'))
final_model.add(Dense(2, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
final_model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


# In[23]:


checkpointer = ModelCheckpoint(filepath=WEIGHTS_FILEPATH, monitor='val_acc', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_acc', verbose=1, patience=500)

#input one data is work...
batch_size = np.shape(sensor_X(train_sensor_val,1,min_len))[0]
#full batch
nb_epoch = 10000
final_model.fit(train_x, train_y,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          callbacks=[early_stopping,checkpointer],
          validation_data=[test_x,test_y]
          )


# In[24]:


from keras.models import model_from_json
# serialize model to JSON
model_json = final_model.to_json()
with open(MODEL_ARCH_FILEPATH, "w") as json_file:
    json_file.write(model_json)
# returns a compiled model
# identical to the previous one


# In[25]:


from keras.models import model_from_json

# load json and create model
json_file = open(MODEL_ARCH_FILEPATH, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(WEIGHTS_FILEPATH)
loaded_model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
print("Loaded model from disk")


# In[26]:


pred = loaded_model.predict_classes(test_x)
# use wafer index to get "voteing"
def pred_label(pred,sensor_val,min_len):
    res_pred = []
    idx = each_len(sensor_val,min_len)
    head = 0
    end = idx[0]
    all_idx = []
    for i in range(0,len(idx)):
        fault_count = pred[range(head,end)].tolist().count(0)
        normal_count = pred[range(head,end)].tolist().count(1)
        all_idx.append(range(head,end))
        if fault_count > normal_count:
            print("number",i,"predict","fault")
            res_pred.append(0)
            if i == len(idx)-1:
                print("stop")
            else:
                head = head + idx[i]
                end = head+idx[i+1]
        elif normal_count > fault_count:
            print("number",i,"predict","normal")
            res_pred.append(1)
            if i == len(idx)-1:
                print("stop")
            else:
                head = head + idx[i]
                end = head+idx[i+1]
        elif normal_count == fault_count:
            print("number",i,"predict","???")
            res_pred.append(0)
            if i == len(idx)-1:
                print("stop")
            else:
                head = head + idx[i]
                end = head+idx[i+1]
    return [np.array(res_pred),all_idx]
real_pred = pred_label(pred,test_sensor_val,149)[0]
all_idx = pred_label(pred,test_sensor_val,149)[1]


# In[27]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_label.astype('float'), real_pred)
print(cm)
TP = np.float32(cm[0,0])
TN = np.float32(cm[1,1])
FP = np.float32(cm[1,0])
FN = np.float32(cm[0,1])
print("TP=",TP)
print("TN=",TN)
print("FP=",FP)
print("FN=",FN)
print("precision=",TP/(TP+FP))
print("recall=",TP/(TP+FN))
print("accuracy=",(TP+TN)/(TP+TN+FP+FN))
