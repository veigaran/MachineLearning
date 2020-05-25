#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2020-05-18 20:39
@Author:Veigar
@File: Pre_process.py
@Github:https://github.com/veigaran
"""


class Preprocess:
    def __init__(self):
        self.start_c = {}  # 开始概率，就是一个字典，state:chance=Word/lines
        self.transport_c = {}  # 转移概率，是字典：字典，state:{state:num,state:num....}   num=num(state1)/num(statess)
        self.emit_c = {}  # 发射概率，也是一个字典，state:{word:num,word,num}  num=num(word)/num(words)
        self.Count_dic = {}  # 一个属性下的所有单词，为了求解emit
        self.state_list = ['Ag', 'a', 'ad', 'an', 'Bg', 'b', 'c', 'Dg',
                           'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l',
                           'Mg', 'm', 'Ng', 'n', 'nr', 'ns', 'nt', 'nx',
                           'nz', 'o', 'p', 'q', 'Rg', 'r', 's', 'na',
                           'Tg', 't', 'u', 'Vg', 'v', 'vd', 'vn', 'vvn',
                           'w', 'Yg', 'y', 'z']
        self.lineCount = -1  # 句子总数，为了求出开始概率
        # 初始化
        for state0 in self.state_list:
            self.transport_c[state0] = {}
            for state1 in self.state_list:
                self.transport_c[state0][state1] = 0.0
            self.emit_c[state0] = {}
            self.start_c[state0] = 0.0
        self.vocabs = []  # 存储词汇
        self.classify = []  # 存储类别
        self.class_count = {}  # 类别对应的个数
        for state in self.state_list:
            self.class_count[state] = 0.0

    def process(self, path):
        with open(path) as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                self.lineCount += 1  # 应该在有内容的行处加 1
                words = line.split(" ")  # 分解为多个单词
                for word in words:
                    position = word.index('/')  # 如果是[中国人民/n]
                    if '[' in word and ']' in word:
                        self.vocabs.append(word[1:position])
                        self.vocabs.append(word[position + 1:-1])
                        break
                    if '[' in word:
                        self.vocabs.append(word[1:position])
                        self.classify.append(word[position + 1:])
                        break
                    if ']' in word:
                        self.vocabs.append(word[:position])
                        self.classify.append(word[position + 1:-1])
                        break
                    self.vocabs.append(word[:position])
                    self.classify.append(word[position + 1:])

                if len(self.vocabs) != len(self.classify):
                    print('词汇数量与类别数量不一致')
                    break  # 不一致退出程序
                    # start_c = {}  # 开始概率，就是一个字典，state:chance=Word/lines
                    # transport_c = {}  # 转移概率，是字典：字典，state:{state:num,state:num....}   num=num(state1)/num(statess)
                    # emit_c = {}  # 发射概率，也是一个字典，state:{word:num,word,num}  num=num(word)/num(words)
                else:
                    for n in range(0, len(self.vocabs)):
                        self.class_count[self.classify[n]] += 1.0
                        if self.vocabs[n] in self.emit_c[self.classify[n]]:
                            self.emit_c[self.classify[n]][self.vocabs[n]] += 1.0
                        else:
                            self.emit_c[self.classify[n]][self.vocabs[n]] = 1.0
                        if n == 0:
                            self.start_c[self.classify[n]] += 1.0
                        else:
                            self.transport_c[self.classify[n - 1]][self.classify[n]] += 1.0
                self.vocabs = []
                self.classify = []
        for state in self.state_list:
            self.start_c[state] = self.start_c[state] * 1.0 / self.lineCount
            for li in self.emit_c[state]:
                self.emit_c[state][li] = self.emit_c[state][li] / self.class_count[state]
            for li in self.transport_c[state]:
                self.transport_c[state][li] = self.transport_c[state][li] / self.class_count[state]
        return self.start_c, self.emit_c, self.transport_c

    # 写入数据到txt
    @staticmethod
    def write2txt(out_path, sentences):
        file = open(out_path, "w", encoding='utf-8')
        for i in sentences:
            file.write(i)
        file.close()
        print("写入成功！")


if __name__ == '__main__':
    handler = Preprocess()
    txt_path = r'./data/corpus_POS.txt'
    start_txt_path = r'./data/start.txt'
    tran_txt_path = r'./data/tran.txt'
    emit_txt_path = r'./data/emit.txt'
    start_c, emit_c, transport_c = handler.process(txt_path)
    handler.write2txt(start_txt_path, str(start_c))
    handler.write2txt(tran_txt_path, str(transport_c))
    handler.write2txt(emit_txt_path, str(emit_c))
