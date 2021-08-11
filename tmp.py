# -*- coding: utf-8 -*-
# @Time     : 2021/6/16 14:20
# @Author   : 宁星星
# @Email    : shenzimin0@gmail.com
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        #查找最长公共子序列的一个最长公共序列，并非返回长度就行
        def pending_text(text1, text2):
            m, n = len(text1), len(text2)
            c = [[0] * (n + 1) for _ in range(m + 1)]
            #用b数组记录查找位置标志
            b = [[""] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    #如果这两个字符相等，则两个子字符串值 + 1
                    if text1[i - 1] == text2[j - 1]:
                        c[i][j] = c[i - 1][j - 1] + 1
                        #共同字符标志
                        b[i][j] = "flag"
                    #如果大于c[i - 1][j] >= c[i][j - 1]，更新为c[i - 1][j]
                    elif c[i - 1][j] >= c[i][j - 1]:
                        c[i][j] = c[i-1][j]
                        #向上查找标志
                        b[i][j] = 'up'
                    #否则更新为c[i][j - 1]
                    else:
                        c[i][j] = c[i][j-1]
                        #向左查找标志
                        b[i][j] = 'left'
            #返回最长公共子序列最长长度(为了后面的代码运行，我这里注释掉了这行))
            # return c[m][n]
            #返回最长公共子序列一个最长公共序列,result作为记录最长子序列字符串，初始化为""
            def print_LCS(b, text, i, j, result):
                #想象成0-m，0-n的网格图，起点在右下角，终点是左上角
                # 如果遍历完成其中一个字符串，则返回result为最长子序列字符串
                if i == 0 or j == 0:
                    return result
                #当在表项中遇到一个'flag'时，意味着是LCS的一个元素,加入result
                if b[i][j] == 'flag':
                    return print_LCS(b, text, i-1, j-1, text[i-1] + result)
                #然后根据之前的过程，反序遍历字符串即可
                elif b[i][j] == 'up':
                    #向上查找
                    return print_LCS(b, text, i-1, j, result)
                else:
                    #向左查找
                    return print_LCS(b, text, i, j - 1, result)
            return print_LCS(b, text1, m, n, "")
        #这里返回该最长公共子序列的长度就是该最长公共子序列问题的解
        # return len(pending_text(text1, text2))
        #这是返回该最长公共子序列字符串
        return pending_text(text1, text2)

class Solution_2:
    def fib(self, n: int) -> int:
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a % 1000000007


if __name__ == '__main__':
    # solution = Solution()
    # print(solution.longestCommonSubsequence("你好，我是中国人", "你好啊，我来自中国，中国人"))
    solution = Solution_2()
    print(solution.fib(100))