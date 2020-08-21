import os
import argparse

parser = argparse.ArgumentParser(description='Compatible Product Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--mix_threshold', type=int, default=4, metavar='N',
                    help='from which epoch to apply semihard')
parser.add_argument('--embed_input', dest='embed_input', action='store_true', default=False,
                    help='whether to use color and pattern embeddings as initial embeddings')
parser.add_argument('--use_updated_embedding', dest='use_updated_embedding', action='store_true', default=False,
                    help='whether to use updated color and pattern embeddings as initial embeddings')
parser.add_argument('--hard_exclusion', dest='hard_exclusion', action='store_true', default=False,
                    help='whether to exclude hard negatives that might be false negatives')
parser.add_argument('--hard_mining', dest='hard_mining', action='store_true', default=False,
                    help='whether to use initial embeddings as auxiliary embeddings to select hard negatives')
parser.add_argument('--batch_mining', default='normal', type=str,
                    help='whether to use average (normal), hard, semihard or a mixture of normal and semihard for triplet loss')
parser.add_argument('--n_masks', type=int, default=5, metavar='N',
                    help='# of subspaces to use')
parser.add_argument('--cond_input', default='none', type=str,
                    help="which input to use for attention weights module. 'same' will use product initial embedding. 'single' will use product category onehot embedding. 'none' will set attention weights to average weights")
parser.add_argument('--triplet_input', dest='triplet_input', action='store_true', default=False,
                    help='whether to sample one negative for each training positive pair')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='which gpu to use')
parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help='To only run inference on test set')
parser.add_argument('--exp_id', default='20200816-032313-306329', type=str,
                    help='name of the trained model')
parser.add_argument('--dim_embed', type=int, default=64, metavar='N',
                    help='how many dimensions in embedding (default: 64)')
parser.add_argument('--l2_embed', dest='l2_embed', action='store_true', default=False,
                    help='L2 normalize the output')
parser.add_argument('--finetune_all', dest='finetune_all', action='store_true', default=False,
                    help='whether to finetune all, including backbone')
parser.add_argument('--margin', type=float, default=0.1, metavar='M',
                    help='margin for triplet loss (default: 0.1)')
parser.add_argument('--eval_interval', type=int, default=200, metavar='N',
                    help='evaluate on valid batches every 200 iters (default 200)')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

import tempfile
import tensorflow as tf
import tensorflow_addons as tfa
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataset import *
from triplet_loss import *
from embedding_model import *
import datetime

# Set up random seeds
np.set_printoptions(precision=4)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
base_iters = 0



def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.001)):
    """ Add kernel regularizations to layers in model
    Args:
        model: embedding model
        regularizer: tf.keras.regularizers.l2(0.001)
    Return:
        model: embedding model
    """
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model
    for layer in model.layers:
        for attr in ['kernel_regularizer','bias_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)
    
    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model

def build_model(args):
    """ build model configurations based on parsed arguments
    Args:
        args: parsed arguments
    Return:
        model: embedding model
    """
    IMG_SHAPE = (224,224,3)
    if args.embed_input:
        model = EmbeddingHead(args)
    else:
        base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE,
                                                     include_top=False,
                                                     weights='imagenet')
        base_model = add_regularization(base_model, regularizer=tf.keras.regularizers.l2(0.0001))
        model = EmbeddingModel(base_model,args)
    return model

def write_args_to_file(args):
    """ Record input argument to logs.txt
    Args:
        args: parsed arguments
    Return:
        exp_name: string
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    f = open('logs.txt','a')
    f.write(str(current_time)+' , '+str(args)+'\n')
    f.close()
    exp_name = str(current_time)
    return exp_name

def tensorboard_setup(exp_name,args):
    """ Setting up tensorboard
    Args:
        exp_name: string
        args: parsed_arguments
    Return:
        summary_writer
    """
    log_dir = 'logs/gradient_tape/' + exp_name
    summary_writer = tf.summary.create_file_writer(log_dir)
    return summary_writer

def calc_set_id_mask(item_id_batch,item2outfit_dict):
    """ Calculate 2d binary outfit mask using outfit_ids
        mask[i,j] = 1 iff i and j ever appeared in one outfit
    Args:
        item_id_batch: list of item ids
        item2outfit_dict: python dict mapping each item to all outfits that contains that item
    Return:
        set_id_mask: tf.bool 'Tensor'
    """
    set_id_mask = np.zeros((item_id_batch.shape[0],item_id_batch.shape[0]))
    item_ids = item_id_batch.numpy()
    for i in range(item_ids.shape[0]):
        for j in range(item_ids.shape[0]):
            # check if i and j have ever appeared in a same outfit
            set_id_mask[i,j] = len(item2outfit_dict[item_ids[i]] & item2outfit_dict[item_ids[j]])>0
    set_id_mask = tf.convert_to_tensor(set_id_mask,dtype=tf.bool)
    return set_id_mask

def train_step_triplet_input(model,args,valid_batches,optimizer,margin,summary_writer,current_epoch):
    """ Train model for one epoch with triplet training data
        Each training positive pair has one negative product
    Args:
        model: embedding model
        args: parsed_arguments
        valide_batch: batched validation dataset
        optimizer: optimizer
        margin: margin for triplet loss
        summary_writer: tensorboard summary writer
        current_epoch: int, epoch number
    """
    global base_iters
    train_batches,item2outfit_dict = get_batch_dataset('train',args)
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_fraction = tf.keras.metrics.Mean('train_fraction', dtype=tf.float32)
    train_dist = tf.keras.metrics.Mean('train_dist', dtype=tf.float32)

    # iterate through training set
    for anchors, positives, negatives, extra_info in train_batches:
        base_iters +=1
        pos_fraction = None
        mean_dist = None
        with tf.GradientTape() as tape:
            image_batch = tf.concat([anchors, positives, negatives], 0)
            # convert label to one-hot embeddings
            anchor_onehot = tf.one_hot(extra_info[:,3],12)
            posneg_onehot = tf.one_hot(extra_info[:,4],12)
            label_batch = tf.concat([tf.concat([anchor_onehot, posneg_onehot, posneg_onehot], 0),
                                     tf.concat([posneg_onehot, anchor_onehot, anchor_onehot], 0)],axis=1)
            
            embeddings = model(image_batch,label_batch)
            
            # calcuate pairwise distance d(a,n) and d(a,p) as well as triplet loss 
            pairwise_dist = tfa.losses.metric_learning.pairwise_distance(embeddings)
            dist_ap = tf.linalg.diag_part(pairwise_dist[:anchors.shape[0],anchors.shape[0]:2*anchors.shape[0]])
            dist_an = tf.linalg.diag_part(pairwise_dist[:anchors.shape[0],2*anchors.shape[0]:3*anchors.shape[0]])
            loss,pos_fraction,mean_dist = triplet_loss_by_pairs(dist_ap,dist_an,margin)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_fraction(pos_fraction)
        train_dist(mean_dist)

        if base_iters % 10 == 0:
            # record loss, positive_fraction and mean_dist
            with summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=base_iters)
                tf.summary.scalar('train_fraction', train_fraction.result(), step=base_iters)
                tf.summary.scalar('train_dist', train_dist.result(), step=base_iters)
            print (base_iters,loss.numpy())
            train_loss.reset_states()
            train_fraction.reset_states()
            train_dist.reset_states()

        if base_iters % args.eval_interval == 0:
            valid_step_triplet_input(model,valid_batches,optimizer,margin,summary_writer)



def train_step(model,args,valid_batches,optimizer,margin,summary_writer,current_epoch):
    """ Train model for one epoch with batched training data
        Each training positive pair has multiple negative products
    Args:
        model: embedding model
        args: parsed_arguments
        valide_batch: batched validation dataset
        optimizer: optimizer
        margin: margin for triplet loss
        summary_writer: tensorboard summary writer
        current_epoch: int, epoch number
    """
    global base_iters
    train_batches,item2outfit_dict = get_batch_dataset('train',args)
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_fraction = tf.keras.metrics.Mean('train_fraction', dtype=tf.float32)
    train_dist = tf.keras.metrics.Mean('train_dist', dtype=tf.float32)
    
    # iterate through training set
    for image_batch, item_id_batch, set_id_batch, cate_id_batch in train_batches:
        base_iters +=1
        pos_fraction = None
        mean_dist = None
        with tf.GradientTape() as tape:
            # set_id_mask[i,j]=True iff i,j are from same set
            set_id_mask = calc_set_id_mask(item_id_batch,item2outfit_dict)
            cate_one_hot = tf.one_hot(cate_id_batch,12)
            embeddings = model(image_batch,cate_one_hot)
            
            if args.batch_mining == 'hard':
                loss,pos_fraction,mean_dist,masks = batch_hard_triplet_loss_v2(
                    set_id_mask, cate_id_batch, embeddings, margin, squared=False)
            elif args.batch_mining == 'normal':
                loss,pos_fraction,mean_dist,masks = batch_all_triplet_loss_v2(
                    set_id_mask, cate_id_batch, embeddings, margin, squared=False)
            elif args.batch_mining == 'semihard':
                loss,pos_fraction,mean_dist,masks = batch_semihard_triplet_loss_v2(
                    set_id_mask, cate_id_batch, embeddings, margin, squared=False)
            elif args.batch_mining == 'semihard_first':
                if current_epoch < args.mix_threshold:
                    loss,pos_fraction,mean_dist,masks = batch_semihard_triplet_loss_v2(
                        set_id_mask, cate_id_batch, embeddings, margin, squared=False)
                else:
                    loss,pos_fraction,mean_dist,masks = batch_all_triplet_loss_v2(
                        set_id_mask, cate_id_batch, embeddings, margin, squared=False)
            elif args.batch_mining == 'semihard_later':
                if epoch_i >= args.mix_threshold:
                    loss,pos_fraction,mean_dist,masks = batch_semihard_triplet_loss_v2(
                        set_id_mask, cate_id_batch, embeddings, margin, squared=False)
                else:
                    loss,pos_fraction,mean_dist,masks = batch_all_triplet_loss_v2(
                        set_id_mask, cate_id_batch, embeddings, margin, squared=False)
            else:
                assert False

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_fraction(pos_fraction)
        train_dist(mean_dist)

        if base_iters % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=base_iters)
                tf.summary.scalar('train_fraction', train_fraction.result(), step=base_iters)
                tf.summary.scalar('train_dist', train_dist.result(), step=base_iters)
                
            print (base_iters,loss.numpy())
            train_loss.reset_states()
            train_fraction.reset_states()
            train_dist.reset_states()

        if base_iters % args.eval_interval == 0:
            valid_step_triplet_input(model,valid_batches,optimizer,margin,summary_writer)

            
def valid_step_triplet_input(model,valid_batches,optimizer,margin,summary_writer):
    """ Test model on validation data
        Each positive pair has one negative products
    Args:
        model: embedding model
        valide_batch: batched validation dataset
        optimizer: optimizer
        margin: margin for triplet loss
        summary_writer: tensorboard summary writer
    """
    global base_iters
    valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)
    valid_fraction = tf.keras.metrics.Mean('valid_fraction', dtype=tf.float32)
    valid_dist = tf.keras.metrics.Mean('valid_dist', dtype=tf.float32)
    valid_batches.shuffle(1000)
    for anchors, positives, negatives, extra_info in valid_batches.take(30):
        image_batch = tf.concat([anchors, positives, negatives], 0)
        # convert label to one-hot embeddings
        anchor_onehot = tf.one_hot(extra_info[:,3],12)
        posneg_onehot = tf.one_hot(extra_info[:,4],12)
        label_batch = tf.concat([tf.concat([anchor_onehot, posneg_onehot, posneg_onehot], 0),
                                 tf.concat([posneg_onehot, anchor_onehot, anchor_onehot], 0)],axis=1)
        embeddings = model(image_batch,label_batch)
        # calcuate pairwise distance d(a,n) and d(a,p) as well as triplet loss 
        pairwise_dist = tfa.losses.metric_learning.pairwise_distance(embeddings)
        dist_ap = tf.linalg.diag_part(pairwise_dist[:anchors.shape[0],anchors.shape[0]:2*anchors.shape[0]])
        dist_an = tf.linalg.diag_part(pairwise_dist[:anchors.shape[0],2*anchors.shape[0]:3*anchors.shape[0]])
        loss,pos_fraction,mean_dist = triplet_loss_by_pairs(dist_ap,dist_an,margin)
        valid_loss(loss)
        valid_fraction(pos_fraction)
        valid_dist(mean_dist)

    with summary_writer.as_default():
        tf.summary.scalar('valid_loss', valid_loss.result(), step=base_iters)
        tf.summary.scalar('valid_fraction', valid_fraction.result(), step=base_iters)
        tf.summary.scalar('valid_dist', valid_dist.result(), step=base_iters)
            
# def valid_step(model,valid_batches,optimizer,margin,summary_writer):
#     global base_iters
#     valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)
#     valid_fraction = tf.keras.metrics.Mean('valid_fraction', dtype=tf.float32)
#     valid_dist = tf.keras.metrics.Mean('valid_dist', dtype=tf.float32)
#     valid_disthp = tf.keras.metrics.Mean('valid_disthp', dtype=tf.float32)
#     valid_disthn = tf.keras.metrics.Mean('valid_disthn', dtype=tf.float32)
#     valid_batches.shuffle(1000)
#     for image_batch, item_id_batch, set_id_batch, cate_id_batch in valid_batches.take(30):
#         pos_fraction = None
#         mean_dist = None
#         embeddings = model(image_batch)
# #         loss,dist_hp,dist_hn = batch_hard_triplet_loss(cate_id_batch, embeddings, margin, squared=False)
# #         loss,pos_fraction,mean_dist = batch_all_triplet_loss(set_id_batch, embeddings, margin, squared=False)
#         if args.tfa_triplet_loss:
#             loss = tfa.losses.triplet_semihard_loss(set_id_batch,embeddings,margin=margin,distance_metric='L2')
#         else:
#             loss,pos_fraction,mean_dist,masks = batch_all_triplet_loss(set_id_batch, cate_id_batch, embeddings, margin, squared=False)
#         valid_loss(loss)
#         if not pos_fraction == None:
#             valid_fraction(pos_fraction)
#         if not mean_dist == None:
#             valid_dist(mean_dist)
# #         valid_disthp(dist_hp)
# #         valid_disthn(dist_hn)

#     with summary_writer.as_default():
#         tf.summary.scalar('valid_loss', valid_loss.result(), step=base_iters)
#         tf.summary.scalar('valid_fraction', valid_fraction.result(), step=base_iters)
#         tf.summary.scalar('valid_dist', valid_dist.result(), step=base_iters)
#         tf.summary.scalar('valid_disthp', valid_disthp.result(), step=base_iters)
#         tf.summary.scalar('valid_disthn', valid_disthn.result(), step=base_iters)
            
def train(args):
    """ Train embedding model with parameters specified from arguments
    Args:
        args: parsed arguments
    """
    exp_name = write_args_to_file(args)
    valid_batches,item2outfit_dict = get_batch_dataset('valid',args)
    model = build_model(args)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    summary_writer = tensorboard_setup(exp_name, args)
    
    # Train model for {args.epochs} epochs
    for epoch_i in range(args.epochs):
        print ('epoch: %d' % epoch_i)
        if args.triplet_input:
            train_step_triplet_input(model,args,valid_batches,optimizer,args.margin,summary_writer,epoch_i)
        else:
            train_step(model,args,valid_batches,optimizer,args.margin,summary_writer,epoch_i)
        check_point_file_name = './checkpoints/'+exp_name+'_epoch_'+str(epoch_i)
        model.save_weights(check_point_file_name)

def triplet_accuracy_triplet_input(model,batches,margin=0):
    """ Test model on provide data batches
    Args:
        model: embedding model
        batches: batches on which the model will be tested
    Return:
        res_acc: Triplet accuracy
    """
    triplet_acc = tf.keras.metrics.Mean('triplet_acc', dtype=tf.float32)
    for anchors, positives, negatives, extra_info in batches.take(50):
        image_batch = tf.concat([anchors, positives, negatives], 0)
        # convert label to one-hot embeddings
        anchor_onehot = tf.one_hot(extra_info[:,3],12)
        posneg_onehot = tf.one_hot(extra_info[:,4],12)
        label_batch = tf.concat([tf.concat([anchor_onehot, posneg_onehot, posneg_onehot], 0),
                                 tf.concat([posneg_onehot, anchor_onehot, anchor_onehot], 0)],axis=1)
        embeddings = model(image_batch,label_batch)
        # calcuate pairwise distance d(a,n) and d(a,p) as well as triplet loss
        pairwise_dist = tfa.losses.metric_learning.pairwise_distance(embeddings)
        dist_ap = tf.linalg.diag_part(pairwise_dist[:args.batch_size,args.batch_size:2*args.batch_size])
        dist_an = tf.linalg.diag_part(pairwise_dist[:args.batch_size,2*args.batch_size:3*args.batch_size])
        loss,pos_fraction,mean_dist = triplet_loss_by_pairs(dist_ap,dist_an,margin)
        triplet_acc(1-pos_fraction)
    print (triplet_acc.result())
    res_acc = triplet_acc.result()
    return res_acc



def test(args):
    """ Test given model on test dataset and output test triplet accuracy
    Args:
        args: parsed arguments
    """
    accuracy = []
    model = build_model(args)

    # load model weights from saved weight file
    model.load_weights('./checkpoints/'+args.exp_id+'_epoch_'+str(19))
    
    # triplet accuracy
#     test_batches,item2outfit_dict = get_batch_dataset('test',args)
#     acc = triplet_accuracy_triplet_input(model,test_batches)

    # compatibility prediction test AUC
    auc = comp_prediction_test(args,model)
    print ('compatibility prediction test AUC:', auc)
    
    # fill-in-the-blank accuracy
    acc = fitb_test(args,model)
    print ('fill-in-the-blank accuracy:',acc)


if args.test:
    test(args)
else:
    train(args)