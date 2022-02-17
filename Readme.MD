# CCKS-2021-事件抽取 A榜第六 方案及代码分享

面向金融领域的篇章级事件抽取和事件因果关系抽取 第六名 方案及代码

## 环境安装
> 依赖文件路径code/conda.txt  和  code/pip.txt <br/>
> 1.conda创建python==3.8环境以及依赖包:  conda env create -f conda.txt <br/>
> 2.Pip安装依赖包： pip install -r pip.txt <br/>


## 数据
训练集数据路径：data/train/ccks_task1_train.txt <br/>
测试集数据路径：data/dev/ccks_task1_eval_data.txt <br/>

### 相关说明
### 1.chinese_roberta_wwm_large_ext_pytorch 预训练模型文件路径 chinese_roberta_wwm_ext_pytorch/
> 下载链接： <br/>
链接：https://pan.baidu.com/s/1UaySHYACs1QOu9Q6cx4MvQ <br/>
提取码：bfd1 <br/>



### 2.投票预测：采用投票基于kfold模型进行事件的集成预测，生成结果文件：系统之神与我同在_valid_result.txt(存放路径为conf.model_folder/系统之神与我同在_valid_result.txt)
