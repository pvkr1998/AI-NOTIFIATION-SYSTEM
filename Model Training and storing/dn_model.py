import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from pandas import ExcelWriter
from pandas import ExcelFile
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

#---------------------------------------------------------Reading and Preprocessing-----------------------------
#start reading from columns
df = pd.read_csv('abc.csv', sep=",", skiprows=2)
df.head()
N = len(df.index)
M = len(df.columns)

cols = df.columns
print("Column headings:")
print(cols[:6])
X = df.iloc[:, :-1]
Y = df.iloc[:, -1:]
enc = X.iloc[:, :6]


#encoding the attributes with categorical values instead of numeric values and 
#remvove original attributes using drop_first after encoded attributes are added to dataframe  
enc.head()
enc = pd.get_dummies(enc, columns=cols[:6], drop_first=True)

#using one hot encoding for ouput
ohe = OneHotEncoder(sparse=False,categories='auto')
Y = pd.DataFrame(ohe.fit_transform(Y))

#Y = pd.get_dummies(Y, columns=Y.columns, drop_first=True)  wont work since Y has only 0 and 1 i.e only 2 diff type of values . so ohe is used
print(len(Y.columns),"= Ycols ",Y[:])
X.drop(df.iloc[:, :6], inplace=True, axis=1)
X.head()
X = pd.concat([enc, X], axis=1)
X.head()
# print(len(X.columns))
X.shape
# X.head()

#total no of columns in input data after encoding
isize = len(X.columns)


#########################################################################################################

#input layer name is "input". giving names to input and output tensors is must to save the model
x = tf.placeholder('float', [None, isize], name="input")
y = tf.placeholder('float')
#no of nodes in each of 3 hidden layers in deep network
hl1 = 400
hl2 = 400
hl3 = 400

#size of output layer
outl = 2

#size of batch. After calculating output of each batch backpropagation or optimisation takes place
batch = 3

#--------------------------------------------building the structre of deep network-------------------
#activation function for hidden layers: relu
def nnmodel(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([isize, hl1])),
    'biases': tf.Variable(tf.random_normal([hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([hl1, hl2])),
    'biases': tf.Variable(tf.random_normal([hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([hl2, hl3])),
    'biases': tf.Variable(tf.random_normal([hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([hl3, outl])),
    'biases': tf.Variable(tf.random_normal([outl])), }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    print("pred=",output)
    return output


def train(x):
    prediction = nnmodel(x)
    #prediction = tf.identity(prediction, name="output")

	#loss functions  
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
	
	#Adam Optimizer is used for adaptive learning rates for each parameter individually
	#we can give learning rate as input also if we want it to be same for all parameters
    optimizer = tf.train.AdamOptimizer().minimize(cost)
	
	#to get probabilities of outputs use softmax function on the output values
    oup = tf.nn.softmax(prediction)
    oup = tf.identity(oup, name="output")#contains probabilities of outputs . it is the final output layer in graph and name is "output"

	#no of times dataset is read
    epochs = 100
	
	#scope of session starts
    with tf.Session() as sess:
		#initializing all the variables
        sess.run(tf.global_variables_initializer())

        print('Total Epochs:', epochs)

        for epoch in range(epochs):
            epoch_loss = 0
            pos = 0
            while pos < N:
                epoch_x = X[pos:pos + batch]
                epoch_y = Y[pos:pos + batch]


				#running the session to calculate ouputs and optimize the parameters
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				
				#adding loss after each batch processing to get total loss after each epoch
                epoch_loss += c
                pos += batch

            print('Epoch', epoch+1, 'loss:', epoch_loss)
			
		#printing training accuracy
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: X, y: Y}))
		
		#saving the graph structre
        tf.train.write_graph(sess.graph_def, '.', 'out/model.pbtxt')
        saver=tf.train.Saver()
		
		#saving parameters in checkpoint file
        saver.save(sess,save_path = "out/model.ckpt")
        
######################################## exporting model to a pb file ##############################################################

#requires names of input and output layers in the network. 
#create a folder named "out"  in the same directory of this file before running this file to get all saved files in "out" folder.
        freeze_graph.freeze_graph(input_graph = "out/model.pbtxt",  input_saver = "",
             input_binary = False, input_checkpoint = "out/model.ckpt", output_node_names = "output",
             restore_op_name = "save/restore_all", filename_tensor_name = "save/Const:0",
             output_graph = "out/frozen_model.pb", clear_devices = True, initializer_nodes = "")
			
        input_graph_def = tf.GraphDef()
        with tf.gfile.Open("out/frozen_model.pb", "rb") as f:
            data = f.read()
            input_graph_def.ParseFromString(data)
			#optimizing the graph data to save memory
            output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def, ["input"],  ["output"], tf.float32.as_datatype_enum)

        f = tf.gfile.GFile("out/model.pb", "w")#storing graph structres and biases in model.pb file
		#writing to pb file
        f.write(output_graph_def.SerializeToString())
#start training
train(x)
