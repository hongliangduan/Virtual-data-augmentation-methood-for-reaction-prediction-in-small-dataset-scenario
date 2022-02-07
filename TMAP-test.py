from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
from mapmaker import map_makers
import os
def combinefile(a,b=None):
##

#input: string a ,b        a： rectan file path b： product file path
#如果只有参数a则代表 a反应的SMILES码

#output: list combinefile ：SMILES_REACTION , int token : 该反应个数

##
   if os.path.exists(b):
        with open(a,'r') as rectans,open(b,'r') as products:

            qian,hou=rectans.readlines(),products.readlines()#按行读入反应、生成物

            assert len(qian)==len(hou),'反应物条目数与生成物条目数不相同'#判断文档的规范性

            qian=[line.strip('\n')for line in qian]#去除每行的换行符来满足反应SMILES码的规范
            hou =[line.strip('\n')for line in hou]
        return [x + '>>' + y for x, y in zip(qian, hou)], len(qian)  ##返回标准的SMILESREACTION
   else:
       with open(a,'r') as SMILES:
            R_SMILES=SMILES.readlines()
            R_SMILES=[line.strip('\n') for line in R_SMILES]
       return R_SMILES,len(R_SMILES)



if __name__ == '__main__':
    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
    fps = []  ##初始化变量 fps：化学反应指纹
    names=[
#         ['tmapfile/bv.source', 'tmapfile/bv.target'],
# ['tmapfile/heck.source', 'tmapfile/heck.target'],
# ['tmapfile/rx3.source', 'tmapfile/rx3.target'],
# ['tmapfile/rx9.source', 'tmapfile/rx9.target'],
        ['augmented_Buchwald-Hartwig', ' '],
        ['Buchwald-Hartwig', ' '],
        ['augmented_Chan-Lam', ' '],
        ['Chan-Lam', ' '],
        ['augmented_Hiyama', ' '],
        ['Hiyama', ' '],
        ['augmented_Kumada', ' '],
        ['Kumada', ' '],
        ['augmented_Suzuki', ' '],
        ['Suzuki', ' '],
    ]##数据输入格式[[反应1，产物]，[反应2,产物],[反应3,产物]........]
    R_SMILES=[]
    R_SIZE=[]
    R_FPS=[]
    for i in names:
        a,b=combinefile(i[0],i[1])
        R_SMILES=R_SMILES+a ##连接所有反应SMILES
        R_SIZE.append(b)    ##获取各个反应数
    for a in range(0,len(R_SMILES),600):
        R_FPS=R_FPS+rxnfp_generator.convert_batch(R_SMILES[a:a+600])
        print(len(R_FPS), len(R_FPS[0]))
    del a,b,i,model,rxnfp_generator,tokenizer,fps,RXNBERTFingerprintGenerator#清理无关变量
    print('stop')
    map_makers(R_SIZE,names,R_SMILES,R_FPS)
