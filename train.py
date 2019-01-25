# -*- coding: utf-8 -*-


from __future__ import print_function
 
import argparse
from random import shuffle
import random
import os
import sys
import math
import tensorflow as tf
import glob
import cv2
 
from image_reader import *
from net import *
from collections import OrderedDict
from keras.optimizers import Adam


parser = argparse.ArgumentParser(description='')
 
parser.add_argument("--snapshot_dir", default='./snapshots', help="path of snapshots") #保存模型的路径
parser.add_argument("--out_dir", default='./train_out', help="path of train outputs") #训练时保存可视化输出的路径
parser.add_argument("--image_size", type=int, default=256, help="load image size") #网络输入的尺度
parser.add_argument("--random_seed", type=int, default=1234, help="random seed") #随机数种子
parser.add_argument('--base_lr', type=float, default=0.0002, help='initial learning rate for adam') #学习率
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')  #训练的epoch数量
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam') #adam优化器的beta1参数
parser.add_argument("--summary_pred_every", type=int, default=200, help="times to summary.") #训练中每过多少step保存训练日志(记录一下loss值)
parser.add_argument("--write_pred_every", type=int, default=100, help="times to write.") #训练中每过多少step保存可视化结果
parser.add_argument("--save_pred_every", type=int, default=5000, help="times to save.") #训练中每过多少step保存模型(可训练参数)
parser.add_argument("--lamda_l1_weight", type=float, default=0.0, help="L1 lamda") #训练中L1_Loss前的乘数
parser.add_argument("--lamda_gan_weight", type=float, default=1.0, help="GAN lamda") #训练中GAN_Loss前的乘数
parser.add_argument("--train_picture_format", default='.png', help="format of training datas.") #网络训练输入的图片的格式(图片在CGAN中被当做条件)
parser.add_argument("--train_label_format", default='.PNG', help="format of training labels.") #网络训练输入的标签的格式(标签在CGAN中被当做真样本)
parser.add_argument("--train_picture_path", default='./layout_collected/', help="path of training datas.") #网络训练输入的图片路径
parser.add_argument("--train_label_path", default='./source_collected/', help="path of training labels.") #网络训练输入的标签路径
#Ying
# use unrolling GAN to avoid model collapse
parser.add_argument("--unrolling_steps", type = int, default=5, help="unrolling steps")#define the unrolling steps
 
args = parser.parse_args() #用来解析命令行参数
EPS = 1e-12 #EPS用于保证log函数里面的参数大于零
 
def save(saver, sess, logdir, step): #保存模型的save函数
   model_name = 'model' #保存的模型名前缀
   checkpoint_path = os.path.join(logdir, model_name) #模型的保存路径与名称
   if not os.path.exists(logdir): #如果路径不存在即创建
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step) #保存模型
   print('The checkpoint has been created.')
 
def cv_inv_proc(img): #cv_inv_proc函数将读取图片时归一化的图片还原成原图
    img_rgb = (img + 1.) * 127.5
    return img_rgb.astype(np.float32) #返回bgr格式的图像，方便cv2写图像
 
def get_write_picture(picture, gen_label, label, height, width): #get_write_picture函数得到训练过程中的可视化结果
    picture_image = cv_inv_proc(picture) #还原输入的图像
    gen_label_image = cv_inv_proc(gen_label[0]) #还原生成的样本
    label_image = cv_inv_proc(label) #还原真实的样本(标签)
    inv_picture_image = cv2.resize(picture_image, (width, height)) #还原图像的尺寸
    inv_gen_label_image = cv2.resize(gen_label_image, (width, height)) #还原生成的样本的尺寸
    inv_label_image = cv2.resize(label_image, (width, height)) #还原真实的样本的尺寸
    output = np.concatenate((inv_picture_image, inv_gen_label_image, inv_label_image), axis=1) #把他们拼起来
    return output
    
def l1_loss(src, dst): #定义l1_loss
    return tf.reduce_mean(tf.abs(src - dst))
 
#Ying 
#unrolled GAN
def extract_update_dict(update_ops):
    """Extract variables and their new values from Assign and AssignAdd ops.
    
    Args:
        update_ops: list of Assign and AssignAdd ops, typically computed using Keras' opt.get_updates()

    Returns:
        dict mapping from variable values to their updated value
    """
    name_to_var = {v.name: v for v in tf.global_variables()}
    updates = OrderedDict()
    for update in update_ops:
        var_name = update.op.inputs[0].name
        var = name_to_var[var_name]
        value = update.op.inputs[1]
        if update.op.type == 'Assign':
            updates[var.value()] = value
        elif update.op.type == 'AssignAdd':
            updates[var.value()] = var + value
        else:
            raise ValueError("Update op type (%s) must be of type Assign or AssignAdd"%update_op.op.type)
    return updates

_graph_replace = tf.contrib.graph_editor.graph_replace

def remove_original_op_attributes(graph):
    """Remove _original_op attribute from all operations in a graph."""
    for op in graph.get_operations():
        op._original_op = None
        
def graph_replace(*args, **kwargs):
    """Monkey patch graph_replace so that it works with TF 1.0"""
    remove_original_op_attributes(tf.get_default_graph())
    return _graph_replace(*args, **kwargs)

def main(): #训练程序的主函数
    if not os.path.exists(args.snapshot_dir): #如果保存模型参数的文件夹不存在则创建
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.out_dir): #如果保存训练中可视化输出的文件夹不存在则创建
        os.makedirs(args.out_dir)
    #train_picture_list = glob.glob(os.path.join(args.train_picture_path, "*")) #得到训练输入图像路径名称列表
    train_picture_list = glob.glob(os.path.join(args.train_label_path, "*")) #得到训练输入图像路径名称列表
    print('train_picture_num =',len(train_picture_list))
    # use label to get layout, because we have more layout figures
    tf.set_random_seed(args.random_seed) #初始一下随机数
    train_picture = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size, 3],name='train_picture') #输入的训练图像
    train_label = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size, 3],name='train_label') #输入的与训练图像匹配的标签
 
    gen_label = generator(image=train_picture, gf_dim=64, reuse=False, name='generator') #得到生成器的输出
    dis_real = discriminator(image=train_picture, targets=train_label, df_dim=64, reuse=False, name="discriminator") #判别器返回的对真实标签的判别结果
    dis_fake = discriminator(image=train_picture, targets=gen_label, df_dim=64, reuse=True, name="discriminator") #判别器返回的对生成(虚假的)标签判别结果
 
    gen_loss_GAN = tf.reduce_mean(-tf.log(dis_fake + EPS)) #计算生成器损失中的GAN_loss部分
    gen_loss_L1 = tf.reduce_mean(l1_loss(gen_label, train_label)) #计算生成器损失中的L1_loss部分
    gen_loss = gen_loss_GAN * args.lamda_gan_weight + gen_loss_L1 * args.lamda_l1_weight #计算生成器的loss
 
    dis_loss = tf.reduce_mean(-(tf.log(dis_real + EPS) + tf.log(1 - dis_fake + EPS))) #计算判别器的loss
 
    gen_loss_sum = tf.summary.scalar("gen_loss", gen_loss) #记录生成器loss的日志
    dis_loss_sum = tf.summary.scalar("dis_loss", dis_loss) #记录判别器loss的日志

    

 
    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph()) #日志记录器
 
    g_vars = [v for v in tf.trainable_variables() if 'generator' in v.name] #所有生成器的可训练参数
    d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name] #所有判别器的可训练参数
 
    #d_optim = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1) #判别器训练器
    #g_optim = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1) #生成器训练器
 
    #updates = d_optim.get_updates(d_vars,[],dis_loss)
    
    #d_grads_and_vars = d_optim.compute_gradients(dis_loss, var_list=d_vars) #计算判别器参数梯度
    #d_train = d_optim.apply_gradients(d_grads_and_vars) #更新判别器参数
    
    # Vanilla discriminator update
    d_opt = Adam(lr=args.base_lr, beta_1=args.beta1)
    updates = d_opt.get_updates(d_vars, [], dis_loss)
    d_train_op = tf.group(*updates, name="d_train_op")

    #updates = d_optim.get_updates(d_vars,[],dis_loss)
    #g_grads_and_vars = g_optim.compute_gradients(gen_loss, var_list=g_vars) #计算生成器参数梯度
    #g_train = g_optim.apply_gradients(g_grads_and_vars) #更新生成器参数
 
    #train_op = tf.group(d_train, g_train) #train_op表示了参数更新操作
    
    #Ying
    #unrolled GAN
    if args.unrolling_steps > 0:
        # get dictionary mapping from variables to their update value after one optimization step
        update_dict = extract_update_dict(updates)
        cur_update_dict = update_dict
        for i in xrange(args.unrolling_steps - 1):
            # Compute variable updates given the previous iteration's updated variable
            cur_update_dict = graph_replace(update_dict, cur_update_dict)
            # Final unrolled loss uses the parameters at the last time step
        unrolled_loss = graph_replace(dis_loss, cur_update_dict)
    else:
        unrolled_loss = dis_loss

    # Optimize the generator on the unrolled loss
    #g_train_opt = tf.train.AdamOptimizer(params['gen_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon'])
    #g_train_op = g_train_opt.minimize(-unrolled_loss, var_list=gen_vars)
    g_optim = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1) #生成器训练器
    g_grads_and_vars = g_optim.compute_gradients(-unrolled_loss, var_list=g_vars) #计算生成器参数梯度
    g_train = g_optim.apply_gradients(g_grads_and_vars) #更新生成器参数
    train_op = tf.group(d_train_op, g_train) #train_op表示了参数更新操作
 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #设定显存不超量使用
    sess = tf.Session(config=config) #新建会话层
    init = tf.global_variables_initializer() #参数初始化器
 
    sess.run(init) #初始化所有可训练参数
 
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=50) #模型保存器
 
    counter = 0 #counter记录训练步数
    #Ying
    fs = [] #record loss
 
    for epoch in range(args.epoch): #训练epoch数
        shuffle(train_picture_list) #每训练一个epoch，就打乱一下输入的顺序
        for step in range(len(train_picture_list)): #每个训练epoch中的训练step数
            counter += 1
            picture_name, _ = os.path.splitext(os.path.basename(train_picture_list[step])) #获取不包含路径和格式的输入图片名称
	    #读取一张训练图片，一张训练标签，以及相应的高和宽
            picture_resize, label_resize, picture_height, picture_width = ImageReader(file_name=picture_name, picture_path=args.train_picture_path, label_path=args.train_label_path, picture_format = args.train_picture_format, label_format = args.train_label_format, size = args.image_size)
            batch_picture = np.expand_dims(np.array(picture_resize).astype(np.float32), axis = 0) #填充维度
            batch_label = np.expand_dims(np.array(label_resize).astype(np.float32), axis = 0) #填充维度
            feed_dict = { train_picture : batch_picture, train_label : batch_label } #构造feed_dict
            gen_loss_value, dis_loss_value, _ = sess.run([gen_loss, dis_loss, train_op], feed_dict=feed_dict) #得到每个step中的生成器和判别器loss
            if counter % args.save_pred_every == 0: #每过save_pred_every次保存模型
                save(saver, sess, args.snapshot_dir, counter)
            if counter % args.summary_pred_every == 0: #每过summary_pred_every次保存训练日志
                gen_loss_sum_value, discriminator_sum_value = sess.run([gen_loss_sum, dis_loss_sum], feed_dict=feed_dict)
                summary_writer.add_summary(gen_loss_sum_value, counter)
                summary_writer.add_summary(discriminator_sum_value, counter)
                #Ying
                #log to record loss is useless, try a new way
                fs.append([counter,gen_loss_sum_value,discriminator_sum_value])

            if counter % args.write_pred_every == 0: #每过write_pred_every次写一下训练的可视化结果
                gen_label_value = sess.run(gen_label, feed_dict=feed_dict) #run出生成器的输出
                write_image = get_write_picture(picture_resize, gen_label_value, label_resize, picture_height, picture_width) #得到训练的可视化结果
                write_image_name = args.out_dir + "/out"+ str(counter) + ".png" #待保存的训练可视化结果路径与名称
                cv2.imwrite(write_image_name, write_image) #保存训练的可视化结果
            print('epoch {:d} step {:d} \t gen_loss = {:.3f}, dis_loss = {:.3f}'.format(epoch, step, gen_loss_value, dis_loss_value))
    #Ying
    #save loss to get graph
    loss_save = './loss_sum.txt'
    np.savetxt(loss_save,fs)

    
if __name__ == '__main__':
    main()
