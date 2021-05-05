
import os
import sys
import time
import re
import math

import numpy as np
import pandas as pd
import tensorflow as tf

# https://www.stefanocappellini.it/tf-name_scope-vs-variable_scope/
#
#

print("""
# 规则1: name_scope只会影响通过 非tf.get_variable 构造的变量, 不会影响通过 get_variable 构造的变量
""")
tf.reset_default_graph()
with tf.name_scope("first"):
    with tf.name_scope("second"):
        v1 = tf.constant(2, name="cons")
        v2 = tf.Variable(2, name="var")
        v3 = tf.multiply(tf.constant(2), tf.constant(3), name="multi")
        v4 = tf.get_variable("get_v", [1])  # 不会有first/second/前缀
print(v1.name, v2.name, v3.name, v4.name, sep='\n')
# first/second/cons:0
# first/second/var:0
# first/second/multi:0
# get_v:0


#
#
#
#
#
#
#

print("""
# 规则2: 默认情况下，重新打开同一个name_scope(name参数相同),会帮你创建一个新的name_scope, 名字是在之前的基础上加上_n
# 比如之前是first, 再次打开就是first_1.
# 如果想要避免这种情况, 可以在再次打开相同scope时带上/, 但是此时就会在变量名字后面带上后缀_n
(1): name_scope中以 / 结尾的名字都是完整的名字，不会在前面加上前缀
(2): "" 或者 None 作为name_scope的名字参数, name_scope会被重置为顶层name_scope
(3): 其他情况下, 当前name_scope名字作为后缀添加进去, 如果已经存在了, 则再在后面加上后缀_n
""")
tf.reset_default_graph()
with tf.name_scope("first"):
    v1 = tf.constant(2, name="cons")
    v2 = tf.Variable(2, name="var")
    v3 = tf.multiply(tf.constant(2), tf.constant(3), name="multi")
    v4 = tf.get_variable("get_v", [1])  # 不会有first/second/前缀

with tf.name_scope("first"):
    v11 = tf.constant(2, name="cons")
    v21 = tf.Variable(2, name="var")
    v31 = tf.multiply(tf.constant(2), tf.constant(3), name="multi")
    v41 = tf.get_variable("get_v1", [1])  # 不会受到name_scope影响
print(v1.name, v2.name, v3.name, v4.name, sep='\n')
print(v11.name, v21.name, v31.name, v41.name, sep='\n')
# first/cons:0
# first/var:0
# first/multi:0
# get_v:0

# first_1/cons:0
# first_1/var:0
# first_1/multi:0
# get_v1:0

print("----------------------------------------------------------")

tf.reset_default_graph()
with tf.name_scope("first"):
    v1 = tf.constant(2, name="cons")
    v2 = tf.Variable(2, name="var")
    v3 = tf.multiply(tf.constant(2), tf.constant(3), name="multi")
    v4 = tf.get_variable("get_v", [1])  # 不会有first/second/前缀
# 虽然此时再次打开的name_scope是一样的了,但是仍然不是同一个变量,tf会把下面的Variable变量加上后缀_n
with tf.name_scope("first/"):
    v11 = tf.constant(2, name="cons")
    v21 = tf.Variable(2, name="var")
    v31 = tf.multiply(tf.constant(2), tf.constant(3), name="multi")
    v41 = tf.get_variable("get_v1", [1])  # 不会受到name_scope影响,因此没有first/前缀
# 这里v411没有name_scope, 名字也为get_v1,看起来等同于v41,其实不是, 通过tf.Variable创建的变量
# 如果有相同name的变量存在,则tf会自动在name参数后面带上后缀_n
v411 = tf.Variable(2, name="get_v1")
print(v1.name, v2.name, v3.name, v4.name, sep='\n')
print(v11.name, v21.name, v31.name, v41.name, sep='\n')
print(v1 == v11, v2 == v21, v3 == v31, v4 == v41)
print(v411.name, v41.name, v411 == v41)
# first/cons:0
# first/var:0
# first/multi:0
# get_v:0

# first/cons_1:0
# first/var_1:0
# first/multi_1:0
# get_v1:0

# False False False False
# get_v1_1:0 get_v1:0 False


print("""
# 规则3: variable_scope被get_variable使用, 且再次打开同一个variable_scope,默认情况下是复用的
,不像name_scope默认不复用(需要以/结尾)
""")
tf.reset_default_graph()
with tf.variable_scope("first"):
    with tf.variable_scope("second"):
        # first/second/get_v:0
        print(tf.get_variable("get_v", [1]).name)

tf.reset_default_graph()
with tf.variable_scope("first"):
    pass
with tf.variable_scope("first"):
    # first/get_variable:0
    print(tf.get_variable("get_variable", [1]).name)

print("""
规则4： 默认情况下：创建variable_scope时会同时创建一个同名的name_scope(多次打开时默认行为不一样)
    (规则1， name_scope不影响 get_variable创建的变量)
    所以,如果多次打开同一个variable_scope,默认情况下第一次的name_scope和variable_scope是一样名字
        但是第二次过后,会在名字后面加上后缀_n(参考规则2)
    但是我们可以设置 auxiliary_name_scope = False 阻止默认行为. 
        或者添加name_scope以 / 结尾 "" None作为参数也可以
""")
tf.reset_default_graph()
with tf.variable_scope("first"):
    v1 = tf.constant(2, name="cons")
    v2 = tf.Variable(2, name="var")
    v3 = tf.multiply(tf.constant(2), tf.constant(3), name="multi")
    v4 = tf.get_variable("get_v", [1])
print(v1.name, v2.name, v3.name, v4.name, sep='\n')

# first/cons:0
# first/var:0
# first/multi:0
# first/get_v:0
print("----------------------------------------------------------")
tf.reset_default_graph()
with tf.variable_scope("first"):
    pass
with tf.variable_scope("first"):
    v1 = tf.constant(2, name="cons")
    v2 = tf.Variable(2, name="var")
    v3 = tf.multiply(tf.constant(2), tf.constant(3), name="multi")
    v4 = tf.get_variable("get_v", [1])
print(v1.name, v2.name, v3.name, v4.name, sep='\n')
# first_1/cons:0
# first_1/var:0
# first_1/multi:0
# first/get_v:0


tf.reset_default_graph()
with tf.variable_scope("first"):
    pass
with tf.variable_scope("first", auxiliary_name_scope=False):  # 阻止variable_scope自动创建name_scope
    v1 = tf.constant(2, name="cons")
    v2 = tf.Variable(2, name="var")
    v3 = tf.multiply(tf.constant(2), tf.constant(3), name="multi")
    v4 = tf.get_variable("get_v", [1])
print(v1.name, v2.name, v3.name, v4.name, sep='\n')

# cons:0 # 顶层变量
# var:0
# multi:0
# first/get_v:0


print("----------------------------练习题------------------------------")
print("-----------------------------ex1-----------------------------")
tf.reset_default_graph()
with tf.variable_scope("first"):
    # 规则4
    print(tf.constant(2, name="c").name)  # first/c
    print(tf.Variable(2, name="v").name)  # first/v
    print(tf.multiply(tf.constant(2), tf.constant(3), name="m").name)  # first/m
    print(tf.get_variable("g", [1]).name)  # first/g
print("-----------------------------ex2-----------------------------")
tf.reset_default_graph()
with tf.variable_scope("first"):
    pass
with tf.variable_scope("first"):
    # 规则4
    print(tf.constant(2, name="c").name)  # first_1/c
    print(tf.Variable(2, name="v").name)  # first_1/v
    print(tf.multiply(tf.constant(2), tf.constant(3), name="m").name)  # first_1/g
    print(tf.get_variable("g", [1]).name)  # first/g
print("-----------------------------ex3-----------------------------")
tf.reset_default_graph()
with tf.variable_scope("first", auxiliary_name_scope=False):
    print(tf.constant(2, name="c").name)  # c
    print(tf.Variable(2, name="v").name)  # v
    print(tf.multiply(tf.constant(2), tf.constant(3), name="m").name)  # m
    print(tf.get_variable("g", [1]).name)  # first/g
print("-----------------------------ex4-----------------------------")
tf.reset_default_graph()
with tf.variable_scope("first"):
    # 规则4+规则2
    with tf.name_scope("another/"):  # note the trailing slash
        print(tf.constant(2, name="c").name)  # another/c
        print(tf.Variable(2, name="v").name)  # another/v
        print(tf.multiply(tf.constant(2), tf.constant(3), name="m").name)  # another/m
        print(tf.get_variable("g", [1]).name)  # first/g
print("-----------------------------ex5-----------------------------")
tf.reset_default_graph()
with tf.variable_scope("first") as scope:
    pass
with tf.variable_scope(scope, auxiliary_name_scope=False):
    with tf.name_scope(scope.original_name_scope):
        print(tf.constant(2, name="c").name)  # first/c
        print(tf.Variable(2, name="v").name)  # first/v
        print(tf.multiply(tf.constant(2), tf.constant(3), name="m").name)  ## first/m
        print(tf.get_variable("g", [1]).name)  # first/g
print("-----------------------------ex6-----------------------------")
tf.reset_default_graph()
with tf.variable_scope("first"):
    pass
with tf.variable_scope("first", auxiliary_name_scope=False):
    with tf.name_scope("first/"):
        print(tf.constant(2, name="c").name)  # first/c
        print(tf.Variable(2, name="v").name)  # first/v
        print(tf.multiply(tf.constant(2), tf.constant(3), name="m").name)  # first/m
        print(tf.get_variable("g", [1]).name)  # first/g

'''
进行变量复用的方法
method1:
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
  scope.reuse_variables()
  output2 = my_image_filter(input2)
method2:
with tf.variable_scope("model"):
  output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
  output2 = my_image_filter(input2)

'''


