import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class EmbeddingModel(tf.keras.Model):
    def __init__(self,base_model,args):
        """ Embedding model that transforms input images into product compatible embeddings
        base_model: backbone model
        args: parsed arguments
        """
        super(EmbeddingModel, self).__init__()
        self.base_model = base_model
        self.embedding_size = args.dim_embed
        if not args.finetune_all:
            self.base_model.trainable = False
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.prediction_layer = tf.keras.layers.Dense(self.embedding_size,
                                                      kernel_initializer='glorot_normal',
                                                      kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                                      bias_regularizer=tf.keras.regularizers.l2(0.001))
        self.args = args
        
    def call(self, inputs,labels=None):
        features = self.base_model(inputs)
        feature_avg = self.global_average_layer(features)
        embeddings = self.prediction_layer(feature_avg)
        if self.args.l2_embed:
            embeddings = tf.nn.l2_normalize(embeddings,axis=1)
        return embeddings
    
class ConditionHead(tf.keras.Model):
    def __init__(self,args):
        """ Attention weights module that learns attention weights to integrate different subspaces
        args: parsed_arguments
        """
        super(ConditionHead,self).__init__()
        self.n_masks = args.n_masks
        self.fc1 = tf.keras.layers.Dense(128,
                                         kernel_initializer='glorot_normal',
                                         kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                         bias_regularizer=tf.keras.regularizers.l2(0.0001))
        self.fc2 = tf.keras.layers.Dense(self.n_masks,
                                         activation='softmax',
                                         kernel_initializer='glorot_normal',
                                         kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                         bias_regularizer=tf.keras.regularizers.l2(0.0001))
    def call(self,inputs):
        weights = self.fc1(inputs)
        weights = self.fc2(weights)
        return weights
    
class EmbeddingHead(tf.keras.Model):
    def __init__(self,args):
        """ Embedding model that transforms initial product embeddings into final product compatible embeddings
        args: parsed arguments
        """
        super(EmbeddingHead, self).__init__()
        self.embedding_size = args.dim_embed
        self.projection_layer = tf.keras.layers.Dense(self.embedding_size,
                                                      kernel_initializer='glorot_normal',
                                                      kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                                      bias_regularizer=tf.keras.regularizers.l2(0.0001))
        self.args = args
        self.n_masks = args.n_masks
        self.mask_list = []
        for i in range(self.n_masks):
            self.mask_list.append(tf.keras.layers.Dense(self.embedding_size,
                                                      kernel_initializer='glorot_normal',
                                                      kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                                      bias_regularizer=tf.keras.regularizers.l2(0.0001)))
        self.condition_head = ConditionHead(args)
    
    def call(self,inputs,labels=None):
        if self.args.cond_input == 'same':
            cond_weights = self.condition_head(inputs)
        elif self.args.cond_input == 'single':
            cond_weights = self.condition_head(labels[:,:12])
        elif self.args.cond_input == 'both':
            cond_weights = self.condition_head(labels)
        elif self.args.cond_input == 'none':
            cond_weights = tf.ones((inputs.shape[0],self.n_masks),dtype=tf.float32)/self.n_masks
        else:
            assert False
        # project input embeddings into subspaces
        masked_embeddings=[]
        for mask in self.mask_list:
            masked_embeddings.append(tf.expand_dims(mask(inputs),1))
            
        # integrate embeddings from subspaces with attention weights
        masked_embeddings = tf.concat(masked_embeddings,axis=1)
        masked_embeddings = tf.math.multiply(masked_embeddings,tf.expand_dims(cond_weights,axis=2))
        masked_embeddings = tf.reduce_sum(masked_embeddings,axis=1)
        if self.args.l2_embed:
            masked_embeddings = tf.nn.l2_normalize(masked_embeddings,axis=1)
        return masked_embeddings