训练模式:
./RM-svm train train_set_path(训练集文件夹路径) 默认../train
	 validation validation_set_path(验证集文件夹路径) 默认../validation
数据集文件夹格式参考train文件夹

预测模式:
./RM-svm prediction img_path(图片路径)
	 model model_path(svm模型路径) 默认../model/svm_arms.xml
