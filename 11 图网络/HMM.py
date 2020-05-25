#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2020-05-18 20:50
@Author:Veigar
@File: HMM.py
@Github:https://github.com/veigaran
"""
import os


class HMM:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.start_txt_path = os.path.join(cur_dir, 'data/start.txt')
        self.tran_txt_path = os.path.join(cur_dir, 'data/tran.txt')
        self.emit_txt_path = os.path.join(cur_dir, 'data/emit.txt')

        self.state_list = ['Ag', 'a', 'ad', 'an', 'Bg', 'b', 'c', 'Dg',
                           'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l',
                           'Mg', 'm', 'Ng', 'n', 'nr', 'ns', 'nt', 'nx',
                           'nz', 'o', 'p', 'q', 'Rg', 'r', 's', 'na',
                           'Tg', 't', 'u', 'Vg', 'v', 'vd', 'vn', 'vvn',
                           'w', 'Yg', 'y', 'z']
        self.start_c = eval(open(self.start_txt_path, 'r', encoding='utf').read())
        self.trans_c = eval(open(self.tran_txt_path, 'r', encoding='utf').read())
        self.emit_c = eval(open(self.emit_txt_path, 'r', encoding='utf').read())

    def viterbi(self, obs, states, start_p, trans_p, emit_p):
        """
        :param obs: 可见序列
        :param states: 隐状态
        :param start_p: 开始概率
        :param trans_p: 转换概率
        :param emit_p: 发射概率
        :return: 序列+概率
        """
        path = {}
        V = [{}]  # 记录第几次的概率
        for state in states:
            V[0][state] = start_p[state] * emit_p[state].get(obs[0], 0)
            path[state] = [state]
        for n in range(1, len(obs)):
            V.append({})
            newpath = {}
            for k in states:
                pp, pat = max([(V[n - 1][j] * trans_p[j].get(k, 0) * emit_p[k].get(obs[n], 0), j) for j in states])
                V[n][k] = pp
                newpath[k] = path[pat] + [k]
                # path[k] = path[pat] + [k]#不能一起变，，后面迭代会用到！
            path = newpath
        (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
        return prob, path[state]

    def predict(self, sentence):
        for li in range(0, len(sentence)):
            sentence[li] = sentence[li].split()
        for li in sentence:
            p, out_list = self.viterbi(li, self.state_list, self.start_c, self.trans_c, self.emit_c)
            # print(li)
            # print(out_list)
            c = list(zip(li, out_list))
            print(c)


if __name__ == '__main__':
    hmm = HMM()
    test_str = [u"你们 站立 在",
                u"我 站 在 北京 天安门 上 大声 歌唱",
                u"请 大家 坐下 喝茶",
                u"你 的 名字 是 什么",
                u"今天 天气 特别 好"]
    hmm.predict(test_str)
