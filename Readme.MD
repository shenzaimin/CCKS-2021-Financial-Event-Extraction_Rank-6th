# CCKS-2021-事件抽取 A榜第六 方案及代码分享

面向金融领域的篇章级事件抽取和事件因果关系抽取 第六名 方案及代码

## 环境安装
> 依赖文件路径code/conda.txt  和  code/pip.txt <br/>
> 1.conda创建python==3.8环境以及依赖包:  conda env create -f conda.txt <br/>
> 2.Pip安装依赖包： pip install -r pip.txt <br/>


## 数据
训练集数据路径：data/train/ccks_task1_train.txt <br/>
测试集数据路径：data/dev/ccks_task1_eval_data.txt <br/>

## 运行步骤
### 训练
> python train_roberta_model.py  --dataset data --num_epochs 30 --model_folder saved_model_roberta_split_train_data_base --seed 2021 --device_num 0 --bert_model_dir chinese_roberta_wwm_ext_pytorch --batch_size 9 --learning_rate 5e-5 --gradient_accumulation_steps 1 --type all --train_or_predict 1 --train_dev_split_rate 0.8
### 预测
> python train_roberta_model.py  --dataset data --model_folder saved_model_roberta_split_train_data_base --seed 2021 --device_num 0 --bert_model_dir chinese_roberta_wwm_ext_pytorch --batch_size 9 --type all --train_or_predict 2
### 相关说明
### 1.chinese_roberta_wwm_large_ext_pytorch 预训练模型文件路径 chinese_roberta_wwm_ext_pytorch/
> 下载链接： <br/>
链接：https://pan.baidu.com/s/1jUZksZi2Kb9hymjSTV1m9Q  <br/>
提取密码：bfd1 <br/>