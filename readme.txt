1.执行分类模型：
cd /Mix_ner4test/CCKS-Cls/
sh classification.sh
得到分类结果文件:/Mix_ner4test/CCKS-Cls/test_output/cls_out_single.csv

2.执行事件抽取
cd /Mix_ner4test/
sh aug.sh
根目录得到结果文件 /Mix_ner4test/result.json 