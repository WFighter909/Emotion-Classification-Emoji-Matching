ECE285 Final Project 

Description
************************************************************

Emotion Classification and Emoji Matching 
by Fan Wang, YueKuan Luo, Bolun Yan, Shuyue Weng, Bowen Zhang


requirements
************************************************************
Python >=2.7
Pytorch >=0.2.0
h5py(Data Preprocessing)
sklearn

Code Organization
************************************************************

--Folders:
    data	   		--  Store the original and preprocessed data
    Fer2013_Resnet18_EP250_Final
	...        		--  Store accuracy data log and trained models
    images	   		--  Emojis for result demonstration and face image for classification 
    models	   		--  VGG 11 13 16 19 models and ResNet18 model library
    transforms     		--  Raw image processing function library
    trained_model  		--  Storing the trained models for demo 
  
--Files:
    fer.py         		--  Class for FER2013 dataset
    training.py    		--  Training model in terminal 
    untils.py      		--  Helper functions	
    data_preprocess.py   	--  Preprocess original data set
    Training_online.ipynb  	--  Training model in Jupyter notebook
    Visualize_demo.ipynb   	--  Visualized result demo 
    
   
How to run 
*************************************************************
Step 1: Put FER2013.csv in data folder
Step 2: Run data_training.py in terminal
Step 3: Put the image to be tested in image folder and put the trained_model in 
        trained_model folder
Setp 4: Run Visualize_demo.ipynb


Training Arguement / Parameter setting
*************************************************************
Training code arguments:
--model, type=str, default='VGG19', help='CNN architecture'
--dataset, type=str, default='FER2013', help='dataset Name'
--bs, default=128, type=int, help='batch size'
--lr, default=0.01, type=float, help='learning rate'
--ep, default=250, type=int, help='total epoch'
--cs, default=44, type=int, help='cut size'
--resume, '-r', action='store_true', help='resume from checkpoint'

Generate a folder (i.e. "FER2013_VGG_19_Ep250") to store the trained models:
path = os.path.join(opt.dataset + '_' + opt.model+'_Ep'+str(opt.ep))

Generate a tarining log file:
logfile = str(path) +"/log.csv"

Visulization Arguement / Parameter setting
*************************************************************
Set the name of input picture stored in 'images/'(without '.jpg'):
pic_name = 'test_img'

Choose the model architecture:
net = VGG('VGG19') or Resnet('Resnet18')

Load the corresponding trained model strored in 'trained_models/' :S
checkpoint = torch.load(os.path.join('trained_models','VGG19_PublicTest_model.t7'),map_location = 'cpu')





 