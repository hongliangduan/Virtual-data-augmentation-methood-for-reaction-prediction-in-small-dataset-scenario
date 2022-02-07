import argparse
from rdkit import Chem
import os

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser(description='test Transformer')

    parser.add_argument('--road_dir', action='store', type=str,
                        default='data/do/jiaochayanzheng/not_kuozeng/k=1/',
                        help='data directory.')



    # road = r'/home/zhangchengyun/Desktop/chemtrm_wzp/out/00'
    args = parser.parse_args()
    road_dir = args.road_dir
    list1 = os.listdir(road_dir)
    f1 = open(str(road_dir + '/' + 'test.target'), 'r')
    list2 = f1.readlines()
    f2 = open(str(road_dir + '/' + 'accuracy.txt'), 'w')
    f5 = open(str(road_dir + '/' + 'standard_test.target'), 'w')
    number1 = 0
    number2 = 0
    number3 = 0
    j = 1
    for dirname in list1:
        if '.out' in dirname and 'standard_' not in dirname:
            file1 = str(road_dir + '/' + dirname)
            file2 = str(road_dir + '/standard_' + dirname)
            f3 = open(file1, 'r')
            f4 = open(file2, 'w')
            l1 = f3.readlines()
            for i in range(len(l1)):
                if l1[i] == list2[i]:
                    number1 += 1
                try:
                    mol1 = Chem.MolFromSmiles(l1[i].replace('\n', ''))
                    mol2 = Chem.MolFromSmiles(list2[i].replace('\n', ''))
                    smiles1 = Chem.MolToSmiles(mol1)
                    smiles2 = Chem.MolToSmiles(mol2)
                    smiles3 = smiles1 + '\n'
                    smiles4 = smiles2 + '\n'
                    f4.write(smiles3)
                    f5.write(smiles4)
                    if l1[i] == smiles4:
                        number2 += 1
                    if smiles3 == smiles4:
                        number3 += 1
                except:
                    f4.write('\n')
                    continue
            accuracy1 = number1 / len(list2)
            accuracy2 = number2 / len(list2)
            accuracy3 = number3 / len(list2)
            # print(accuracy1)
            # print(accuracy2)
            print(road_dir)
            print(accuracy3)
            f2.write(str(j) + '、' + str('标准化前' + dirname + '和标准化前的test.target相比准确率是' + str(accuracy1)) + '\n'
                     + str(j) + '、' + '标准化前' + str(dirname) + '和标准化后的test.target相比准确率是' + str(accuracy2) + '\n'
                     + str(j) + '、' + '标准化后' + str(dirname) + '和标准化后的test.target相比准确率是' + str(accuracy3) + '\n')
            number1 = 0
            number2 = 0
            number3 = 0
            j += 1