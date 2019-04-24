import pandas as pd
import numpy as np
import random
import tensorflow as tf

##$data preprocess for PLC

class Data_Preprocess_PLC:

    def __init__(self,plc_path,window_size,batch_size):
        self.plc_path = plc_path
        self.plc = pd.read_csv(plc_path)
        self.csv_no = list(set(pd.read_csv(plc_path)['csv_no'])) 
        self.window_size = window_size
        self.batch_size = batch_size

    def Dict_plc_time(self):
        plc = self.plc
        csv_no = self.csv_no
        plc_feature = list(plc.columns)[1:-1]

        time = list(plc['time'])
        for i in range(len(time)):
            time_s = int(time[i].split(':')[2])
            time_m = int(time[i].split(':')[1])
            time_h = int(time[i].split(':')[0])
            time_all = time_s+(time_m*60)+(time_h*3600)
            time[i] = time_all

        off_day = [0]
        for i in range(len(time)-1):
            if time[i]> time[i+1]:
                off_day.append(time[i]-time[i+1])
            else:
                off_day.append(off_day[i])
        for i in range(len(time)):
            time[i] = time[i]+off_day[i]

        sharp_time = [0]
        for i in range(len(time)-1):
            if time[i+1]-time[i]>600:
                sharp_time.append(time[i+1]-time[i]+sharp_time[i]-300)
            else:
                sharp_time.append(sharp_time[i])

        correct_time = []
        for i in range(len(time)):
            correct_time.append(time[i]-sharp_time[i])

        plc['time'] = correct_time
        dict_plc_time = {}
        for i in range(len(csv_no)):
            plc_time = list(plc[plc['csv_no']==self.csv_no[i]]["time"])
            dict_plc_time[csv_no[i]] = plc_time #$
        #plc_min_max_standard = (plc[plc_feature]-plc[plc_feature].min())/(plc[plc_feature].max()-plc[plc_feature].min())  
        plc_data = plc[plc_feature] 
        #plc_min_max_standard = pd.concat([plc_min_max_standard,plc['csv_no']],axis=1) #$
        plc_data = pd.concat([plc_data,plc['csv_no']],axis=1).fillna(method='ffill') 
        return plc_data,dict_plc_time

    def Overall_Window_index(self):
        plc = self.plc
        csv_no = self.csv_no
        overall_window_index = {}
        for i in range(len(self.csv_no)):
            window_number = int((plc[plc['csv_no']==csv_no[i]].shape[0])/self.window_size)
            #window_number = self.plc[self.plc['csv_no']==self.csv_no[i]].shape[0]-self.window_size+1
            overall_window_index[self.csv_no[i]] = window_number
        return overall_window_index
            
    def Overall_Window_index_shuffle(self):
        overall_window = []
        for i in range(len(self.csv_no)):
            window_number = self.plc[self.plc['csv_no']==self.csv_no[i]].shape[0]-self.window_size+1
            for j in range(window_number):
                overall_window.append((self.csv_no[i],j))
        agent = [i for i in range(len(overall_window))]
        random.shuffle(agent)
        overall_window_index_shuffle = [overall_window[agent[i]] for i in range(len(agent))]
        return overall_window_index_shuffle

    def PLC_window_generator(self,plc_data,dict_plc_time,csv_no_point,window_point):
        plc_data = plc_data[plc_data['csv_no']==csv_no_point]
        plc_data = np.array(plc_data)[:,:4]
        plc_window = plc_data[window_point*self.window_size:(window_point*self.window_size)+self.window_size]
        ## 
        #start_window_time_s = (int(dict_plc_time[csv_no_point][window_point+self.window_size].split(':')[2])+int(dict_plc_time[csv_no_point][window_point].split(':')[2]))/2
        #start_window_time_m = int(dict_plc_time[csv_no_point][window_point+self.window_size].split(':')[1])
        #start_window_time_h = int(dict_plc_time[csv_no_point][window_point+self.window_size].split(':')[0])
        #start_window_time = (start_window_time_h*3600)+(start_window_time_m*60)+start_window_time_s
        start_window_time = (dict_plc_time[csv_no_point][window_point+self.window_size]+dict_plc_time[csv_no_point][window_point])/2
        ##
        end_csv_no_point = list(dict_plc_time.keys())[-1]
        #end_window_time_s = int(dict_plc_time[end_csv_no_point][-1].split(':')[2])
        #end_window_time_m = int(dict_plc_time[end_csv_no_point][-1].split(':')[1])
        #end_window_time_h = int(dict_plc_time[end_csv_no_point][-1].split(':')[0])
        #end_window_time = (end_window_time_h*3600)+(end_window_time_m*60)+end_window_time_s
        end_window_time = dict_plc_time[end_csv_no_point][-1]
        diff_window_time = end_window_time-start_window_time

        return plc_window,diff_window_time  ##csv_no_point 1,2,3... ; window_point 0,1,2,3...

    def Generator_PLC_window(self):
        point = 0
        overall_window_index_shuffle = self.Overall_Window_index_shuffle()  
        while True:
            if point < len(overall_window_index_shuffle):         
                X_batch,Y_batch = [],[]
                for i in [point+j for j in range(self.batch_size)]:
                    csv_no_point = overall_window_index_shuffle[i][0]
                    window_point = overall_window_index_shuffle[i][1]
                    plc_window,diff_window_time = self.PLC_window_generator(csv_no_point,window_point)
                    X_batch.append(plc_window)   
                    Y_batch.append(diff_window_time)
                X_batch = np.array(X_batch)
                Y_batch = np.array(Y_batch)
                point += self.batch_size
            else:
                point = 0
            yield (X_batch,Y_batch)

##$data preprocess for Snesor 

class Data_Preprocess_Sensor:

    def __init__(self,sensor_path,window_size):

        self.sensor_path = sensor_path
        self.window_size = window_size
        if sensor_path != 'Empty':
            self.csv_no_point = int(sensor_path.split('/')[-1].split('.')[0])

    def Sensor_data(self):
        sensor_data = pd.read_csv(self.sensor_path).fillna(method='ffill')
        #sensor_max_min_standard = (sensor-sensor.min())/(sensor.max()-sensor.min())
        return sensor_data

    def Sensor_window_generator(self,sensor_data,overall_window_index,sensor_window_point):
        #sensor_window_size = int((sensor_data.shape[0])/overall_window_index[self.csv_no_point])
        sensor_window_size = self.window_size*755
        sensor_data = np.array(sensor_data)
        if (sensor_window_point*sensor_window_size)+sensor_window_size < sensor_data.shape[0]:
            sensor_window = sensor_data[(sensor_window_point*sensor_window_size):(sensor_window_point*sensor_window_size)+sensor_window_size]
        else:
            sensor_window = np.array([0 for i in range(sensor_data[1]*sensor_window_size)]).reshape((sensor_window_size,sensor_data[1]))
        return sensor_window


class Data_Preprocess:
    
    def __init__(self,plc_path,sensor_path,window_size):
        #window_size = int(input('Please input window size!! :\n'))
        self.window_size = window_size
        self.plc_path = plc_path
        self.sensor_path = sensor_path
        batch_size = 32 #dont touch
        self.data_preprocess_PLC = Data_Preprocess_PLC(plc_path,window_size,batch_size)
        self.data_preprocess_Sensor = Data_Preprocess_Sensor(sensor_path,window_size) 
    
    def index_search(self):
        #print('This is Index Table!!')
        overall_window_index = self.data_preprocess_PLC.Overall_Window_index()
        #window_point = int(input('Please input window point!! (EX: 0,1,2,..22) :\n'))
        return overall_window_index
    
    def Sensor_data(self):
        sensor_data = self.data_preprocess_Sensor.Sensor_data()
        return sensor_data

    def PLC_data(self):
        plc_data,dict_plc_time = self.data_preprocess_PLC.Dict_plc_time()
        return plc_data,dict_plc_time

    def PLC_Window_generator(self,window_point,plc_data,dict_plc_time):      
        csv_no_point = self.data_preprocess_Sensor.csv_no_point
        plc_window,RUL_window = self.data_preprocess_PLC.PLC_window_generator(plc_data,dict_plc_time,csv_no_point,window_point) 
        overall_window_index = self.index_search()
        #sensor_window = self.data_preprocess_Sensor.Sensor_window_generator(sensor_data,overall_window_index,window_point)
        #print('=====================OUTPUT=====================')
        #print('PLC window shape : ',plc_window.shape)
        #print('Sensor window shape : ',sensor_window.shape)
        #print('The RUL of widnow : ',RUL_window)
        return plc_window,RUL_window

    def Sensor_Window_generator(self,window_point,sensor_data):      
        #csv_no_point = self.data_preprocess_Sensor.csv_no_point
        #plc_window,RUL_window = self.data_preprocess_PLC.PLC_window_generator(plc_data,dict_plc_time,csv_no_point,window_point) 
        overall_window_index = self.index_search()
        sensor_window = self.data_preprocess_Sensor.Sensor_window_generator(sensor_data,overall_window_index,window_point)
        return sensor_window


class Data_Structure_Process:

    def __init__(self,plc_1_path,plc_2_path,sensor_1_path,sensor_2_path,window_size):
        self.plc_1_path = plc_1_path
        self.plc_2_path = plc_2_path
        self.sensor_1_path = sensor_1_path
        self.sensor_2_path = sensor_2_path
        self.window_size = window_size

    def Dict_data_map(self):

        data_preprocess_1 = Data_Preprocess(self.plc_1_path,'Empty',self.window_size)
        overall_window_index_1 = data_preprocess_1.index_search()
        data_preprocess_2 = Data_Preprocess(self.plc_2_path,'Empty',self.window_size)
        overall_window_index_2 = data_preprocess_2.index_search()

        index_1_key = list(overall_window_index_1.keys())
        index_2_key = list(overall_window_index_2.keys())
        point_1_num = sum([overall_window_index_1[index_1_key[i]] for i in range(len(index_1_key))])
        point_2_num = sum([overall_window_index_2[index_2_key[i]] for i in range(len(index_2_key))])
        point_num = point_1_num + point_2_num

        dict_data_map = {}
        cumulate_num = 0
        for i in range(len(index_1_key)):
            point_num = overall_window_index_1[index_1_key[i]]
            for j in range(point_num):
                dict_data_map[cumulate_num] = '1/' +str(index_1_key[i])+'/'+str(j)
                cumulate_num +=1
        for i in range(len(index_2_key)):
            point_num = overall_window_index_2[index_2_key[i]]
            for j in range(point_num):
                dict_data_map[cumulate_num] = '2/' +str(index_2_key[i])+'/'+str(j)
                cumulate_num +=1        
        return dict_data_map

    def Index_shuffle(self,dict_data_map):
        data_map_keys = list(dict_data_map.keys())
        random.shuffle(data_map_keys)
        return agent

    def Generator_Sensor_window(self,dict_data_map):
        point = 0
        index_shuffle = self.Index_shuffle(dict_data_map)  
        while True:
            if point < len(index_shuffle):         
                X_batch = []
                for i in [point+j for j in range(self.batch_size)]:
                    index = index_shuffle[i]
                    data_map = dict_data_map[index].split('/')
                    class_no = data_map[0]
                    time_no = data_map[1]
                    window_no = data_map[2]
                    if class_no == '1':
                        data_preprocess_1 = Data_Preprocess(self.plc_1_path,self.sensor_1_path+time_no+'.csv',self.window_size)
                        sensor_data = data_preprocess_1.Sensor_data()
                        sensor_window = data_preprocess_1.Sensor_Window_generator(int(time_no),sensor_data)
                    elif class_no == '2':
                        data_preprocess_2 = Data_Preprocess(self.plc_2_path,self.sensor_2_path+time_no+'.csv',self.window_size)        
                        sensor_data = data_preprocess_2.Sensor_data()
                        sensor_window = data_preprocess_2.Sensor_Window_generator(int(time_no),sensor_data)
                    X_batch.append(sensor_window)   
                X_batch = np.array(X_batch)
                point += self.batch_size
            else:
                point = 0
            yield X_batch


##$$$DEMO

plc_1_path = '/Users/hsucheng/Documents/Statistic_Consult/dataset/01/PLC/plc.csv'
plc_2_path = '/Users/hsucheng/Documents/Statistic_Consult/dataset/03/PLC/plc.csv'
sensor_1_path = '/Users/hsucheng/Documents/Statistic_Consult/dataset/01/Sensor/'
sensor_2_path = '/Users/hsucheng/Documents/Statistic_Consult/dataset/03/Sensor/'







window_size = 10
data_preprocess = Data_Preprocess(plc_1_path,sensor_1_path+'1.csv',window_size)
#overall_window_index = data_preprocess.index_search()
#print(overall_window_index)
plc_data,dict_plc_time = data_preprocess.PLC_data()
sensor_data = data_preprocess.Sensor_data()
plc_window,RUL_window = data_preprocess.PLC_Window_generator(1,plc_data,dict_plc_time)
sensor_window = data_preprocess.Sensor_Window_generator(1,sensor_data)

print('=====================OUTPUT=====================')
print('PLC window shape : ',plc_window.shape)
print('Sensor window shape : ',sensor_window.shape)
print('The RUL of widnow : ',RUL_window)



















###$ Model
##@autoencoder-LSTM

class Autoencoder_LSTM:  
    def __init__(self):

        self.window_size = window_size
        self.batch_size = batch_size
        self.window_dim = 4
        self.embedding_dim = embedding_dim
        LR = 0.002         

    def Framework(self):
        # tf placeholder
        tf_x = tf.placeholder(tf.float32, [None, self.window_size * self.window_dim])      
        
        # encoder
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=64)
        enc_h_t, (enc_h_c, enc_h_n) = tf.nn.dynamic_rnn(rnn_cell,tf_x,initial_state=None,dtype=tf.float32,time_major=False)                                                                                       
        encoded = tf.layers.dense(h_t, self.embedding_dim,tf.nn.tanh)

        # decoder
        dec_h_t, (dec_h_c, dec_h_n) = tf.nn.dynamic_rnn(rnn_cell,encoded,initial_state=None,dtype=tf.float32,time_major=False)                      
        decoded = tf.layers.dense(dec_h_t, [self.window_size,self.window_dim], tf.nn.sigmoid)
        return 

    def Loss_Function(self):
        loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
        train = tf.train.AdamOptimizer(LR).minimize(loss)
        return

    def Optimization():
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())


        for step in range(10):
            b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
            _, encoded_, decoded_, loss_ = sess.run([train, encoded, decoded, loss], {tf_x: b_x})

            if step % 100 == 0:     # plotting
                print('train loss: %.4f' % loss_)
                # plotting decoded image (second row)
                decoded_data = sess.run(decoded, {tf_x: view_data})
                for i in range(N_TEST_IMG):
                    a[1][i].clear()
                    a[1][i].imshow(np.reshape(decoded_data[i], (28, 28)), cmap='gray')
                    a[1][i].set_xticks(()); a[1][i].set_yticks(())
                plt.draw(); plt.pause(0.01)
        plt.ioff()

        # visualize in 3D plot
        view_data = test_x[:200]
        encoded_data = sess.run(encoded, {tf_x: view_data})
        fig = plt.figure(2); ax = Axes3D(fig)
        X, Y, Z = encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2]
        for x, y, z, s in zip(X, Y, Z, test_y):
            c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
        ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
        plt.show()

    def DD(window_size,window_dim):
        input_data = Input(shape=(window_size,window_dim))

        # encoder layers
        enc_lstm_ht = LSTM(100,input_shape=(window_size,window_dim))(input_data) 
        encoded = Dense(64,activation='relu')(enc_lstm_ht)
        encoder_output = Dense(encoding_dim)(encoded)

        # decoder layers
        dec_lstm_ht = LSTM(100,input_shape=(window_size,64))(encoder_output) 
        decoded = Dense(10, activation='relu')(encoder_output)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(784, activation='tanh')(decoded)

        # construct the autoencoder model
        autoencoder = Model(input=input_img, output=decoded)

        # construct the encoder model for plotting
        encoder = Model(input=input_img, output=encoder_output)

        # compile autoencoder
        autoencoder.compile(optimizer='adam', loss='mse')

        # training
        autoencoder.fit(x_train, x_train,
                        epochs=20,
                        batch_size=256,
                        shuffle=True)

        # plotting
        encoded_imgs = encoder.predict(x_test)



