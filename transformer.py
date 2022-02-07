#!/usr/bin/python
# -*- coding: UTF-8 -*-

#==============================================================================
#
#      Description: Training script for transformer
#       Department: NLP
#             Date: 2018-03-05
#            Admin: Carl Cai
#           E-mail: caix@kalamodo.com
#    Copyright2018: Kalamodo Co. Ltd.
#
#==============================================================================

import time
import json
import argparse
from numpy.core.numeric import cross
import tensorflow as tf
import numpy as np
#from sklearn.model_selection import train_test_split
from DeepZone.multigpu.sync import multi_gpus_train_op
from DeepZone.utils.gpuinfo import get_gpu_names
from DeepZone.models import build_model_fn, build_kwargs
# from transformer_multigpu import TransformerMultiGPU
from DeepZone.models.Transformer.transformer import Transformer
from utils import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='training Transformer')
parser.add_argument('--data_dir', action='store', type=str, 
                    default='/home/tanshui/桌面/WXY_transformer/data/kumada/kumada_raw/8/', help='data directory.')
parser.add_argument('--max_epochs', '-max_ep', action='store', type=int,
                    default=200000, help='maximum epochs for training.')
parser.add_argument('--model_dir', action='store', type=str,
                    default='model/kumada/kumada_raw/8/', help='directory to save model')
parser.add_argument('--model_name', action='store', type=str,
                    default='ChemTrm', help='name of model')
parser.add_argument('--approx_num', action='store', type=int, default=3144,
                    help='batch size for training.')


def learning_rate_factor(name, 
                         step_num,
                         learning_rate_constant=2.,
                         learning_rate_warmup_steps=16000,
                         learning_rate_decay_steps=5000,
                         learning_rate_decay_rate=1.,
                         learning_rate_decay_staircase=False,
                         train_steps=2000000,
                         hidden_size=None):
  """Compute the designated learning rate factor from hparams."""
  if name == "constant":
    tf.logging.info("Base learning rate: %f", learning_rate_constant)
    return learning_rate_constant
  elif name == "linear_warmup":
    return tf.minimum(1.0, step_num / learning_rate_warmup_steps)
  elif name == "linear_decay":
    ret = (train_steps - step_num) / learning_rate_decay_steps
    return tf.minimum(1.0, tf.maximum(0.0, ret))
  elif name == "cosdecay":  # openai gpt
    in_warmup = tf.cast(step_num <= learning_rate_warmup_steps,
                        dtype=tf.float32)
    ret = 0.5 * (1 + tf.cos(
        np.pi * step_num / learning_rate_decay_steps))
    # if in warmup stage return 1 else return the decayed value
    return in_warmup * 1 + (1 - in_warmup) * ret
  elif name == "rsqrt_decay":
    return tf.rsqrt(tf.maximum(step_num, learning_rate_warmup_steps))
  elif name == "rsqrt_normalized_decay":
    scale = tf.sqrt(tf.to_float(learning_rate_warmup_steps))
    return scale * tf.rsqrt(tf.maximum(
        step_num, learning_rate_warmup_steps))
  elif name == "exp_decay":
    decay_steps = learning_rate_decay_steps
    warmup_steps = learning_rate_warmup_steps
    p = (step_num - warmup_steps) / decay_steps
    p = tf.maximum(p, 0.)
    if learning_rate_decay_staircase:
      p = tf.floor(p)
    return tf.pow(learning_rate_decay_rate, p)
  elif name == "rsqrt_hidden_size":
    assert hidden_size is not None
    return hidden_size ** -0.5
#   elif name == "legacy":
#     return legacy_learning_rate_schedule(hparams)
  else:
    raise ValueError("unknown learning rate factor %s" % name)


def learning_rate_schedule(schedule_string, step_num, hidden_size=None):
  """Learning rate schedule based on hparams."""
  names = schedule_string.split("*")
  names = [name.strip() for name in names if name.strip()]
  ret = tf.constant(1.0)
  for name in names:
    ret *= learning_rate_factor(name, step_num, hidden_size=hidden_size)
  return ret

old_acc=0
def eval(test_model, 
         sess, 
         writer, tf_acc, acc_summary, global_step,
         val_data, max_length,
         approx_num, 
         code_of_start, 
         code_of_end, 
         code_of_pad
         ,saver):
    np.random.shuffle(val_data)
    N = len(val_data)
    pos = 0
    correct = 0
    approx_num *= 3

    def post_proc(sent_ids):
        out = []
        for idx in sent_ids:
            if idx == code_of_start:
                continue
            elif idx == code_of_end:
                break
            else:
                out.append(idx)
        return out

    def is_same(sent_a, sent_b):
        if len(sent_a) != len(sent_b):
            return False

        for a, b in zip(sent_a, sent_b):
            if a != b:
                return False
        return True

    while True:
        # get batch data
        batch_inputs, batch_targets, new_batch_size, _ = \
                                get_batch_data_approx(val_data, pos, approx_num)
        
        # padd and get position
        # front padding for inputs of encoder
        batch_inputs = batch_pad(batch_inputs, val=code_of_pad, front=False)

        pred_sents = test_model.predict(sess, batch_inputs, max_length)

        ##### calc
        for pred, tgt in zip(pred_sents, batch_targets):
            pred = post_proc(pred)
            tgt = post_proc(tgt)
            correct += 1 if is_same(pred, tgt) else 0

        print('{}/{}'.format(pos+new_batch_size, N))
        if (pos+new_batch_size) >= N:
            break

        # update pos
        pos += new_batch_size
    global old_acc
    acc = correct / float(N)
    if acc > old_acc:
        saver.save(sess, model_dir + model_name, global_step=global_step)
        print("保存了最高点")
        old_acc=acc
    acc_summary_str, gs = sess.run([acc_summary, global_step], {tf_acc: acc})
    writer.add_summary(acc_summary_str, global_step=gs)
    print('eval process done!')


if __name__ == '__main__':
    args = parser.parse_args()

    # get arguments
    data_dir = args.data_dir
    max_epochs = args.max_epochs
    model_dir = args.model_dir
    model_name = args.model_name
    approx_num = args.approx_num

    # get all visible GPUs
    gpu_ids = get_gpu_names(id_only=True)
    gpu_ids = set(gpu_ids)
    num_gpus = len(gpu_ids)
    print('using GPU ids: ', gpu_ids)

    # read data from files
    data, vocab, sign_start, sign_end = read_data(data_dir)
    train_data = data['train']
    val_data = data['dev']
    test_data = data['test']
    all_data = train_data + val_data + test_data
    max_sent_len_in = max(map(len, [sent[0] for sent in all_data]))
    max_sent_len_out = max(map(len, [sent[1] for sent in all_data]))
    # print(max_sent_len_in, max_sent_len_out)
    max_sent_len = max(max_sent_len_in, max_sent_len_out)
    print('the largest sentence size={}'.format(max_sent_len))

    # reform training data and valdata
    #train_data = train_data + val_data
    #train_data, val_data = train_test_split(train_data, test_size=.0)

    print('training data size={}'.format(len(train_data)))
    print('validation data size={}'.format(len(val_data)))
    print('test data size={}'.format(len(test_data)))

    # add sign of pad for label set
    sign_pad = '<pad>'
    # labelset = labelset | set([sign_pad])

    ### load word2idx
    try:
        # with open(model_dir+'/../'+'word2idx.json', 'r') as ifile:
        with open('/home/tanshui/桌面/ChemTrm_all/word2idx55.json', 'r') as ifile:
            json_str = ifile.readline()

        vocab_set = json.loads(json_str)
        print('vocab loaded.')
    except:
        print('vocab not found!!!!')
        exit(-1)

    idx2word_vocab = inverse_dict(vocab_set)
    vocab_size = len(vocab_set)
    print(vocab_set)
    print(idx2word_vocab)
    print('vocabulary size={}'.format(vocab_size))
    # word2idx
    train_data = trans_data(train_data, vocab_set)
    val_data = trans_data(val_data, vocab_set)
    test_data = trans_data(test_data, vocab_set)

    # setup parameters
    n_heads = 8
    emb_dim = 256
    num_layers = 6
    FFN_inner_units = 2048
    dropout_keep_prob = 0.7

    global_step = tf.get_variable('global_step', initializer=tf.constant(0.))
    # calculate learning rate
    # lr = tf.constant(2e-4)
    lr_schedule = 'constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size'
    lr = learning_rate_schedule(
        lr_schedule, global_step, 
        hidden_size=emb_dim
    )
    
    kwargs = build_kwargs(
        voca_size=vocab_size,
        label_size=vocab_size,
        code_of_start=vocab_set[sign_start],
        code_of_end=vocab_set[sign_end],
        code_of_pad=vocab_set[sign_pad],
        num_layers_enc=num_layers,
        num_layers_dec=num_layers,
        emb_dim=emb_dim,
        n_heads=n_heads,
        dropout_keep_prob=dropout_keep_prob,
        FFN_inner_units=FFN_inner_units
    )
    ## optimizer
    opt = tf.contrib.opt.LazyAdamOptimizer(
        lr,
        beta1=0.9,
        beta2=0.997,
        epsilon=1e-9
    )

    # build graph
    def build_model_fn_wrapper(reuse=tf.AUTO_REUSE, is_train=True):
        model = build_model_fn(
            Transformer, **kwargs, reuse=reuse, is_train=is_train,
            lr=lr, opt=opt, global_step=global_step
        )
        return model
    
    # get multigpu train op
    # loss_op, train_op, models = multi_gpus_train_op(
    #     opt=opt, 
    #     build_model_fn=build_model_fn_wrapper, 
    #     gpu_ids=gpu_ids,
    #     global_step=global_step
    # )
    model = build_model_fn_wrapper(is_train=True)
    # model for test
    test_model = build_model_fn_wrapper(is_train=False)

    #### build graph for acc
    tf_acc = tf.placeholder(tf.float32)
    acc_summary = tf.summary.scalar('eval_acc', tf_acc)
    
    # create session

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.InteractiveSession(config=sess_config)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    # set a summary writer
    writer = tf.summary.FileWriter(model_dir, sess.graph)

    # read checkpoint
    ckpt = tf.train.latest_checkpoint(checkpoint_dir=model_dir)
    if ckpt is None:
        sess.run(tf.global_variables_initializer())
        print('model initialized.')
    else:
        saver.restore(sess, ckpt)
        print('latest model -- {} has been loaded.'.format(ckpt))

    # training procedure
    N = len(train_data)
    beam_size = 4
    log_n_steps = 10
    _last_t = -1.
    _curr_t = time.time()
    kaishi = time.time()
    for epoch in range(max_epochs):
        # shuffle training data
        np.random.shuffle(train_data)
        losses = []
        np_lrs = []
        pos = 0
        log_rec = 0
        if time.time() - kaishi > 39600 :
            break
        while True:
            # get batch data
            # batch_inputs, _, batch_targets, _, new_batch_size = \
            #                     get_batch_data(train_data, pos, batch_size)
            batch_inputs, batch_targets, new_batch_size, numTokens = \
                                  get_batch_data_approx(train_data, pos, approx_num)
            
            # padd and get position
            # front padding for inputs of encoder
            batch_inputs = batch_pad(batch_inputs, val=vocab_set[sign_pad], front=False)
            # backward padding for targets (inputs of decoder)
            batch_targets = batch_pad(
                batch_targets,
                val=vocab_set[sign_pad],
                front=False
            )
            
            loss, np_lr, train_summary = model.update(sess, batch_inputs, batch_targets)
            ##########
            # split to all gpus
            # try:
            #     batch_inputs = np.split(batch_inputs, num_gpus)
            #     pos_in = np.split(pos_in, num_gpus)
            #     batch_targets = np.split(batch_targets, num_gpus)
            #     pos_out = np.split(pos_out, num_gpus)
            # except:
            #     break

            # # feed to all gpus
            # feed_dict = {}
            # for i in range(num_gpus):
            #     feed_dict.update({
            #         models[i].inputs: batch_inputs[i],
            #         models[i].pos_in: pos_in[i],
            #         models[i].outputs: batch_targets[i],
            #         models[i].pos_out: pos_out[i]
            #     })
            # # run train op
            # loss, _, gs = sess.run(
            #     [loss_op, train_op, global_step], 
            #     feed_dict=feed_dict
            # )
            losses.append(loss)
            np_lrs.append(np_lr)

            # terminate condition
            # if new_batch_size < batch_size:
            #     break
            if (pos+new_batch_size) >= N:
                break

            # print information
            over_print('{}/{}'.format(pos+new_batch_size, N))
            # if (pos+new_batch_size) % (batch_size*10) == 0:
            if log_rec % log_n_steps == 0:
                over_print(
                    '{}/{}, mean_loss={:.5f}, lr={:.5e}'.format(
                        pos+new_batch_size, N, np.mean(losses), np.mean(np_lrs)
                    )
                )
                writer.add_summary(train_summary, global_step=model.get_global_step(sess))
                print()
                losses = []
                np_lrs = []
                log_rec = 0
            
            # update pos
            pos += new_batch_size
            log_rec += 1
        
        # print last information
        over_print(
            '{}/{}, mean_loss={}'.format(pos+new_batch_size, N, np.mean(losses))
        )
        print()

        ### save model
        # saver.save(sess, model_dir+model_name, global_step=global_step)
        # print('epoch {} finished and model saved!'.format(epoch))

        if time.time()-_curr_t>300:          
            eval(
                test_model,
                sess,
                writer, tf_acc, acc_summary, global_step,
                val_data, max_sent_len, approx_num,
                vocab_set[sign_start], vocab_set[sign_end], vocab_set[sign_pad],saver
            )
            _curr_t = time.time()
