#---------------------------------------------------------------------
#Name: Make_Surrogate_Model.py
#Menu-ja: �㗝���f���쐬(����)
#Menu-en: Create a surrogate model (beta version)
#Type: Python
#Create: June 31, 2021 JSOL Corporation
#Comment-en: Create a surrogate model (NN, SVR, RT).
#Comment-ja: �㗝���f���iNN,SVR,RT�j�̍쐬���s���܂��B
#Copyright: (c) JSOL Corporation. All Rights Reserved.
#---------------------------------------------------------------------
# -*- coding: utf-8 -*-

import csv
import math
import designer
import numpy as np
import pickle
import random
import time
import os
import zipfile
import shutil
import warnings

py_error = 0
app = designer.GetApplication()
# NN param
num_neuron = 20
num_layer = 3
num_epoch = 100
num_batch = 10

warnings.simplefilter('ignore')

def show_message_immediately(title_en, title_jp, message_en, message_jp):
	msgdlg = app.CreateDialogBox()
	msgdlg.SetTranslation(title_en, title_jp)
	msgdlg.SetTranslation(message_en, message_jp)
	msgdlg.SetCancelButtonVisible(False)
	msgdlg.SetTitle(title_en)
	msgdlg.AddLabel(message_en)
	msgdlg.Show()

if (sys.version >= "3.8"):
	try:
		import tensorflow as tf
		from sklearn.preprocessing import StandardScaler
		from sklearn.svm import SVR
		from sklearn.tree import DecisionTreeRegressor
		from sklearn.model_selection import KFold
		from sklearn.metrics import r2_score
		from sklearn.metrics import mean_squared_error
	except:
		title_en = "Python library error"
		title_jp = "Python���C�u�����G���["
		message_en = "The runtime library cannot be found. Please refer to the manual for the required packages."
		message_jp = "�����^�C�����C�u������������܂���B�K�v�ȃp�b�P�[�W�̓}�j���A�����Q�Ƃ��Ă��������B"
		show_message_immediately(title_en, title_jp, message_en, message_jp)
		py_error = 1

if (sys.version < "3.8"):
	title_en = "Python Version Error"
	title_jp = "Python�o�[�W�����G���["
	message_en = "Please use Python of newer version over 3.8"
	message_jp = "�o�[�W����3.8�ȏ��Python���g�p���Ă��������B"
	show_message_immediately(title_en, title_jp, message_en, message_jp)
	py_error = 1

class DialogData:
	def __init__(self):
		self.model_path = ""
		self.csv_path   = ""
		self.model = 1
		self.cross_val = 5
		self.cross_frag = 1
		self.response_name = []
		self.isValid  = False

def main():
	obj_list = []
	param_list = []
	model_save_path = []
	random_seed = 0
	num_case = 0
	RMS = 0.0
	R2 = 0.0
	Error = 0
	random_seed = random.randint(0,2**32-1)
	
	AnalysisGroup  = app.GetCurrentAnalysisGroup()
	if AnalysisGroup.IsValid() == False:
		AnalysisGroup = app.GetCurrentStudy()
	if AnalysisGroup.IsValid() == False:
		message_en = "The study or analysis group cannot be found."
		message_jp = "study�܂��͉�̓O���[�v��������܂���B"
		show_error_exit_message(message_en, message_jp)
		return

	data = get_data_from_input_dialog()
	if data.isValid == False:
		return

	OptTable = AnalysisGroup.GetOptimizationTable()
	num_param = OptTable.NumParameters()	
	num_obj = len(data.response_name)
	kfold_split = data.cross_val
 
	for i in range(num_obj):
		tmp = data.response_name[i]
		obj_list.append(tmp.decode())
		model_save_path.append(data.model_path + b"/" + data.response_name[i] + b"/" + data.response_name[i])
		if not os.path.exists(data.model_path + b"/" + data.response_name[i]):
			os.mkdir(data.model_path + b"/" + data.response_name[i])
		
	for i in range(num_param):
		param = OptTable.GetParametricItem(i)
		if param.GetItemName() != "":
			param_list.append(param.GetItemName())
		else:
			param_list.append(param.GetParameterName())

	input_param,result_correct,Error,learn_data =loadcsv(data.csv_path,param_list,obj_list)
	if Error == 1:
		return

	num_case = len(input_param)

	correct_data = np.zeros([num_case])
	result_out_csv = np.zeros([num_case,num_param + 3])

	if data.model ==1:
		tf.get_logger().setLevel("ERROR")
		ann_tmp = NN_function()

	for i in range(num_obj):
		sc_X = StandardScaler()
		sc_y = StandardScaler()
		for j in range(num_case):
			correct_data[j] = result_correct[j][i]	
		correct_data = correct_data.reshape(num_case, 1)

		tmp_X = sc_X.fit_transform(input_param)
		tmp_y = sc_y.fit_transform(correct_data)
		pickle.dump(sc_X, open(model_save_path[i] + b"_x.pkl", 'wb'))
		pickle.dump(sc_y, open(model_save_path[i] + b"_y.pkl", 'wb'))
		
		start = time.time()
		if data.cross_frag == 1:
			if data.model ==1:
				NN_model(tmp_X,tmp_y,sc_y,model_save_path,i,ann_tmp)
				RMS,R2,Error = NN_KFold(random_seed,tmp_X,tmp_y,sc_y,result_out_csv,num_param,input_param,kfold_split,ann_tmp)
			elif data.model ==2:
				SVR_model(tmp_X,tmp_y,sc_y,model_save_path,i)
				RMS,R2,Error = SVR_KFold(random_seed,tmp_X,tmp_y,sc_y,result_out_csv,num_param,input_param,kfold_split)
			elif data.model ==3:
				RT_model(tmp_X,tmp_y,sc_y,model_save_path,i)
				RMS,R2,Error = RT_KFold(random_seed,tmp_X,tmp_y,sc_y,result_out_csv,num_param,input_param,kfold_split)
		else:
			if data.model ==1:
				NN_model(tmp_X,tmp_y,sc_y,model_save_path,i)
			elif data.model ==2:
				SVR_model(tmp_X,tmp_y,sc_y,model_save_path,i)
			elif data.model ==3:
				RT_model(tmp_X,tmp_y,sc_y,model_save_path,i)	 
		
		end = time.time()
		elapsed_time = end - start
		
		save_csv(model_save_path,result_out_csv,i,num_case,num_param,param_list,elapsed_time,obj_list,RMS,R2,data)
		save_zip(model_save_path,data,i,learn_data)

		if Error:
			message_en = "The response value " + data.response_name[i].decode() + " may contain an outlier. Delete the case that contains an outlier."
			message_jp = "�����l" + data.response_name[i].decode() + "�Ɉُ�l���܂܂�Ă���\��������܂��B�ُ�l���܂܂��P�[�X���폜���Ă��������B"
			show_error_exit_message(message_en, message_jp)
			return
		

	show_normal_exit_message()

def save_zip(model_save_path,data,index,learn_data):
	save_path = data.model_path + b"/" + data.response_name[index] + b"/learn_data.csv"
	save_path = save_path.decode()

	with open(save_path, "a",encoding="cp932") as f:
		for i in range(len(learn_data)):
			for j in range(len(learn_data[0])):
				f.write("\""+learn_data[i][j] + "\"" + ",")
			f.write("\n")

	dirname = data.model_path + b"/" + data.response_name[index]
	dirname = dirname.decode()
	shutil.make_archive(dirname, format='zip', root_dir = data.model_path + b"/" + data.response_name[index])
	shutil.rmtree(dirname)

def NN_function():
	ann = tf.keras.models.Sequential()
	for i in range(num_layer):
		ann.add(tf.keras.layers.Dense(units=num_neuron, activation='relu'))
	ann.add(tf.keras.layers.Dense(units=1, activation='linear'))

	return ann 

def NN_model(tmp_X,tmp_y,sc_y,model_save_path,index,ann_tmp):
	local_y = tmp_y
	local_y = np.reshape(local_y,(-1))
	ann = tf.keras.models.clone_model(ann_tmp)
	ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
	ann.fit(tmp_X,local_y, epochs = num_epoch, batch_size = num_batch, verbose=0)
	ann.save(model_save_path[index] + b".surm")

def NN_KFold(random_seed,tmp_X,tmp_y,sc_y,result_out_csv,num_param,input_param,kfold_split,ann_tmp):
	index_count = 0
	Start_index = 0
	Error = 0
	RMS = 0.0
	R2 = 0.0
	kf = KFold(n_splits=kfold_split, shuffle=True,random_state =random_seed)

	for train_index, test_index in kf.split(tmp_X):
		local_y = tmp_y[train_index, :]
		local_y = np.reshape(local_y,(-1))
		
		ann = tf.keras.models.clone_model(ann_tmp)
		ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
		ann.fit(tmp_X[train_index, :],local_y, epochs = num_epoch, batch_size = num_batch, verbose=0)
		result_predict_tmp = sc_y.inverse_transform(ann.predict(tmp_X[test_index, :]))
		result_correct_tmp = sc_y.inverse_transform(tmp_y[test_index, :])

		try:
			RMS = RMS + np.sqrt(mean_squared_error(result_correct_tmp, result_predict_tmp))
			R2 = R2 + r2_score(result_correct_tmp, result_predict_tmp) 
		except:
			Error = 1

		index_count = index_count + 1
		Start_index_tmp = save_result(test_index,Start_index,index_count,result_predict_tmp,result_correct_tmp,result_out_csv,num_param,input_param)
		Start_index = Start_index_tmp
	return RMS/kfold_split,R2/kfold_split,Error

def SVR_model(tmp_X,tmp_y,sc_y,model_save_path,i):
	regressor_svr = SVR(kernel='rbf')
	local_y = tmp_y
	local_y = np.reshape(local_y,(-1))
	regressor_svr.fit(tmp_X,local_y)
	pickle.dump(regressor_svr, open(model_save_path[i] + b".surm", 'wb'))

def SVR_KFold(random_seed,tmp_X,tmp_y,sc_y,result_out_csv,num_param,input_param,kfold_split):
	index_count = 0
	Start_index = 0
	RMS = 0.0
	R2 = 0.0
	Error = 0
	kf = KFold(n_splits=kfold_split, shuffle=True,random_state =random_seed)

	for train_index, test_index in kf.split(tmp_X):
		local_y = tmp_y[train_index, :]
		local_y = np.reshape(local_y,(-1))
		regressor_svr = SVR(kernel='rbf')
		regressor_svr.fit(tmp_X[train_index, :],local_y)
		result_predict_tmp = sc_y.inverse_transform(regressor_svr.predict(tmp_X[test_index, :]))
		result_correct_tmp = sc_y.inverse_transform(tmp_y[test_index, :])
		
		try:
			RMS = RMS + np.sqrt(mean_squared_error(result_correct_tmp, result_predict_tmp))
			R2 = R2 + r2_score(result_correct_tmp, result_predict_tmp) 
		except:
			Error = 1

		index_count = index_count + 1
		Start_index_tmp = save_result(test_index,Start_index,index_count,result_predict_tmp,result_correct_tmp,result_out_csv,num_param,input_param)
		Start_index = Start_index_tmp
	return RMS/kfold_split,R2/kfold_split,Error

def RT_model(tmp_X,tmp_y,sc_y,model_save_path,i):
	regressor_tree = DecisionTreeRegressor(random_state=0)
	local_y = tmp_y
	local_y = np.reshape(local_y,(-1))
	regressor_tree.fit(tmp_X,local_y)
	pickle.dump(regressor_tree, open(model_save_path[i] + b".surm", 'wb'))

def RT_KFold(random_seed,tmp_X,tmp_y,sc_y,result_out_csv,num_param,input_param,kfold_split):
	index_count = 0
	Start_index = 0
	RMS = 0.0
	R2 = 0.0
	Error = 0

	kf = KFold(n_splits=kfold_split, shuffle=True,random_state =random_seed)
	for train_index, test_index in kf.split(tmp_X):
		local_y = tmp_y[train_index, :]
		local_y = np.reshape(local_y,(-1))
		regressor_tree = DecisionTreeRegressor(random_state=0)
		regressor_tree.fit(tmp_X[train_index, :],local_y)
		result_predict_tmp = sc_y.inverse_transform(regressor_tree.predict(tmp_X[test_index, :]))
		result_correct_tmp = sc_y.inverse_transform(tmp_y[test_index, :])
		
		try:
			RMS = RMS + np.sqrt(mean_squared_error(result_correct_tmp, result_predict_tmp))
			R2 = R2 + r2_score(result_correct_tmp, result_predict_tmp) 
		except:
			Error = 1

		index_count = index_count + 1
		Start_index_tmp = save_result(test_index,Start_index,index_count,result_predict_tmp,result_correct_tmp,result_out_csv,num_param,input_param)
		Start_index = Start_index_tmp
	return RMS/kfold_split,R2/kfold_split,Error

def save_result(test_index,Start_index,index_count,result_predict_tmp,result_correct_tmp,result_out_csv,num_param,input_param):
	for i in range(len(test_index)):
		result_out_csv[Start_index][0] = index_count
		result_out_csv[Start_index][1] = result_correct_tmp[i]
		result_out_csv[Start_index][2] = result_predict_tmp[i]
		for j in range(num_param):
			result_out_csv[Start_index][3+j] = input_param[test_index[i]][j]
		Start_index = Start_index + 1
	return Start_index

def save_csv(model_save_path,result_out_csv,index,num_case,num_param,param_list,elapsed_time,obj_list,RMS,R2,data):
	

	if data.cross_frag == 1:
		save = np.empty(shape=(num_case+7,num_param+5), dtype=np.object)
		for i in range(num_case+7):
			for j in range(num_param+5):
				save[i][j] = ""		
	
		save[0][0] = obj_list[index]
		save[1][0] = "model"
		if data.model ==1:
			save[1][1] = "NN"
		elif data.model ==2:
			save[1][1] = "SVR"
		elif data.model == 3:
			save[1][1] = "RT"
		save[2][0] = "time(s)"
		save[2][1] = str(elapsed_time)
		save[3][0] = "RMS"
		save[3][1] = str(RMS)
		save[4][0] = "R2"
		save[4][1] = str(R2)
		save[6][0] = "case"
		save[6][1] = "K-fold" 
		save[6][2] = "correct"
		save[6][3] = "predict"
		for j in range(num_param):
			save[6][5+j] = param_list[j]

		for i in range(num_case):
			save[i+7][0] = str(i)
			save[i+7][1] = str(result_out_csv[i][0])
			save[i+7][2] = str(result_out_csv[i][1])
			save[i+7][3] = str(result_out_csv[i][2])
			for j in range(num_param):
				save[i+7][5+j] = str(result_out_csv[i][3+j])
	else:
		save = np.empty(shape=(3,2), dtype=np.object)
		for i in range(3):
			for j in range(2):
				save[i][j] = ""		
	
		save[0][0] = obj_list[index]
		save[1][0] = "model"
		if data.model ==1:
			save[1][1] = "NN"
		elif data.model ==2:
			save[1][1] = "SVR"
		elif data.model == 3:
			save[1][1] = "RT"
		save[2][0] = "time(s)"
		save[2][1] = str(elapsed_time)
	
	file_name = data.model_path + b"/" + data.response_name[index] + b".csv"
	file_name = file_name.decode()
	np.savetxt(file_name,save,fmt="%s",delimiter=',')

def loadcsv(csv_path,param_list,obj_list):
	error = 0
	row_offset = 0
	index_param = []
	index_result = []
	input_param = None
	result_correct = None
	with open(csv_path, "r",encoding='cp932') as f:
		for tmp_col in csv.reader(f):
			break;

	col_param = len(param_list)	
	col_correct = len(obj_list)

	if tmp_col[0] == "Objective Values":
		row_offset = 1
		count = 0
		with open(csv_path, "r",encoding='cp932') as f:
			for tmp_col in csv.reader(f):
				if count == 1:
					break;
				count = count + 1
	
	col = len(tmp_col)

	for i in range(col_param):
		for j in range(col):
			if param_list[i] == tmp_col[j]:
				index_param.append(j) 
				break;
			if j == (col-1):
				message_en = "The variable name " + param_list[i] + "�@cannot be found in the csv file."
				message_jp = "�ϐ��� " + param_list[i] + "�@��csvfile���猩����܂���"
				show_error_exit_message(message_en, message_jp)
				error = 1
				return input_param,result_correct,error
	
	for i in range(col_correct):
		for j in range(col):
			if obj_list[i] == tmp_col[j]:
				index_result.append(j) 
				break;
			if j == (col-1):
				message_en = "The variable name " + param_list[i] + "�@cannot be found in the csv file."
				message_jp = "�ϐ��� " + obj_list[i] + "�@��csvfile���猩����܂���"
				show_error_exit_message(message_en, message_jp)
				error = 1
				return input_param,result_correct,error
	
	with open(csv_path, "r",encoding='cp932') as f:
		reader = csv.reader(f)
		string_list = [row for row in reader]
	row = len(string_list)
	
	input_param = np.zeros([row - 1 - row_offset,col_param])
	result_correct = np.zeros([row - 1- row_offset,col_correct])

	for i in range(row_offset,row - 1):
		for j in range(col_param):
			try:
				input_param[i-row_offset][j]=float(string_list[i+1][int(index_param[j])])
			except:
				message_en = "A character string was detected in the " + str(i) + "row and " +  str(j) + "column. Please delete it."
				message_jp = str(i + 2 + row_offset) + "�s�� " + str(j) + "��ڂɕ���������m���܂����B�폜���Ă��������B"
				show_error_exit_message(message_en, message_jp)
				error = 1
			
	for i in range(row_offset,row- 1):
		for j in range(col_correct):
			try:
				result_correct[i - row_offset][j] = float(string_list[i+1][int(index_result[j])])
			except:
				message_en = "A character string was detected in the " + str(i) + "row and " +  str(j) + "column. Please delete it."
				message_jp = str(i + 2 + row_offset) + "�s�� " + str(j) + "��ڂɕ���������m���܂����B�폜���Ă��������B"
				show_error_exit_message(message_en, message_jp)
				error = 1

	learn_data_tmp = []
	for i in range(row_offset,row):
		learn_data_tmp.append(string_list[i]) 

	return input_param,result_correct,error,learn_data_tmp

def get_data_from_input_dialog():
	dialog = create_input_dialog()
	dialog.Show()
	if dialog.WasCancelled() == False:
		data = get_values_from_input_dialog(dialog)
		return data
	return DialogData()

def get_values_from_input_dialog(dialog):
	data = DialogData()
	data.model_path = dialog.GetValue("surrogate_model_path")
	data.csv_path = dialog.GetValue("csv_file_name")
	data.cross_val = dialog.GetValue("cross_val")
	data.model = dialog.GetValue("model")
	data.cross_frag = dialog.GetValue("cross_frag")
	param_tmp = dialog.GetValue("response_name")
	param_tmp = param_tmp.decode()
	param = param_tmp.split(',')
	
	if data.csv_path == b"":
		message_en = "Specify input file name."
		message_jp = "���̓t�@�C�������w�肵�Ă��������B"
		show_error_exit_message(message_en, message_jp)
		return data

	if data.model_path == b"":
		message_en = "Specify output directory name."
		message_jp = "�o�̓t�H���_�����w�肵�Ă��������B"
		show_error_exit_message(message_en, message_jp)
		return data

	if os.path.isfile(data.csv_path) == 0 :
		message_en = "The input file path is incorrect."
		message_jp = "���̓t�@�C���p�X������������܂���B"
		show_error_exit_message(message_en, message_jp)
		return data

	if os.path.isdir(data.model_path) == 0:
		message_en = "The output folder is incorrect."
		message_jp = "�o�̓t�H���_������������܂���B"
		show_error_exit_message(message_en, message_jp)
		return data
	
	if param_tmp == "":
		message_en = "Specify the name of the response value."
		message_jp = "�����l�̖��O���w�肵�Ă��������B"
		show_error_exit_message(message_en, message_jp)
		return data
	
	if data.cross_val <= 1:
		message_en = "Set the number of divisions to 2 or more."
		message_jp = "��������2�ȏ�Őݒ肵�Ă��������B"
		show_error_exit_message(message_en, message_jp)
		return data
	
	for i in range(len(param)):
		Error = 0
		for j in range(len(data.response_name)):
			if param[i].encode() == data.response_name[j]:
				Error = 1
		if Error == 0:
			data.response_name.append(param[i].encode())

	data.isValid = True
	return data

def create_input_dialog():
	app = designer.GetApplication()
	dialog = app.CreateDialogBox()

	title_jp = "�㗝���f���쐬(����)"
	title_en = "Create a surrogate model (beta version)"
	inputlbl_jp = "����:"
	inputlbl_en = "Input:"
	outputlbl_jp = "�o��:"
	outputlbl_en = "Output:"
	csvinput_jp = "�w�K�f�[�^�Z�b�g"
	csvinput_en = "Training dataset" 
	surrogate_path_jp = "�㗝���f���o�̓p�X"
	surrogate_path_en = "Surrogate model output path" 
	cross_val_jp = "K-������������"
	cross_val_en = "K-Fold  Cross validation"	
	fold_val_jp = "������"
	fold_val_en = "K-fold"
	response_jp = "�����l"
	response_en = "Response value"	
	response_name_jp = "�ϐ����i�J���}��؂�j"
	response_name_en = "Variable name(Comma separated)"	
	cross_use_jp = "��������"
	cross_use_en = "Cross validation"
	cross_notuse_jp = "���؂Ȃ�" 
	cross_notuse_en = "No verification"

	dialog.SetTranslation(title_en, title_jp)

	dialog.SetTranslation(response_en, response_jp)
	dialog.SetTranslation(response_name_en, response_name_jp)
	dialog.SetTranslation(fold_val_en, fold_val_jp)
	dialog.SetTranslation(cross_val_en, cross_val_jp)
	dialog.SetTranslation(inputlbl_en, inputlbl_jp)
	dialog.SetTranslation(outputlbl_en, outputlbl_jp)
	dialog.SetTranslation(csvinput_en, csvinput_jp)
	dialog.SetTranslation(surrogate_path_en, surrogate_path_jp)
	dialog.SetTranslation(cross_use_en, cross_use_jp)
	dialog.SetTranslation(cross_notuse_en, cross_notuse_jp)

	dialog.AddLabel(response_en)
	dialog.AddString("response_name", response_name_en, "")
	dialog.AddLine()
	
	dialog.AddLabel(cross_val_en)
	dialog.AddRadio("cross_frag", cross_use_en, 1)
	dialog.AddInteger("cross_val", fold_val_en, 5)
	dialog.AddRadio("cross_frag", cross_notuse_en, 2)
	dialog.AddLine()

	dialog.AddRadio("model", "Neural network", 1)
	dialog.AddRadio("model", "Support Vector Regression", 2)
	dialog.AddRadio("model", "Regression tree", 3)
	dialog.AddLine()

	dialog.SetTitle(title_en)
	dialog.AddLabel(inputlbl_en)
	dialog.AddOpenFilename("csv_file_name", csvinput_en,"","CSV file (*.csv)")
	
	dialog.AddLine()
	dialog.AddLabel(outputlbl_en)
	dialog.AddDirectoryPath("surrogate_model_path", surrogate_path_en,"./")
	dialog.AddLine()

	return dialog

def show_normal_exit_message():
	title_en = "Finished"
	title_jp = "�I��"
	message_en = "Finished."
	message_jp = "�I��"

	show_message(title_en, title_jp, message_en, message_jp)

def show_error_exit_message(message_en, message_jp):
	title_en = "Error"
	title_jp = "�G���["

	show_message(title_en, title_jp, message_en, message_jp)

def show_message(title_en, title_jp, message_en, message_jp):
	msgdlg = app.CreateDialogBox()

	msgdlg.SetTranslation(title_en, title_jp)
	msgdlg.SetTranslation(message_en, message_jp)

	msgdlg.SetCancelButtonVisible(False)
	msgdlg.SetTitle(title_en)
	msgdlg.AddLabel(message_en)
	msgdlg.Show()

if py_error == 0:
	main()

