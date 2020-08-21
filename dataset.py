import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import compatible_product_pb2 as compatible_product
import tensorflow_addons as tfa
import time
import hnswlib
from sklearn.metrics import roc_auc_score, roc_curve

def read_files():
    """ Read necessary json files containing image-path, semantic categories and item metadata
        Return:
            cate_dict: python dict for semantic category mapping
            item_list: python dict for item metadata
    """
    polyvore_img_path = '../vasileva2018_learning/fashion-compatibility-master/data/polyvore_outfits/images/'
    polyvore_img_path = pathlib.Path(polyvore_img_path)

    # prepare category mapping
    with open('../vasileva2018_learning/fashion-compatibility-master/data/polyvore_outfits/categories.csv') as f:
        category_ls  = f.read()
    category_ls = category_ls.strip().split('\n')
    cate_dict = {}
    for line in category_ls:
        line  = line.strip().split(',')
        cate_dict[line[0]] = line[2]
    
    # prepare item metadata
    with open('../vasileva2018_learning/fashion-compatibility-master/data/polyvore_outfits/polyvore_item_metadata.json') as f:
        item_list = json.load(f)

    return cate_dict,item_list

def read_message_from_file():
    """ load color and pattern embedding list from bytes
    """
    image_list = compatible_product.Images()
    f = open('polyvore_outfits.outside_proto.bytes', "rb")
    image_list.ParseFromString(f.read())
    f.close()
    return image_list

def read_updated_message_from_file():
    """ load updated color and pattern embedding list from bytes
    """
    updated_list1 = compatible_product.Images()
    updated_list2 = compatible_product.Images()
    f1 = open('polyvore_outfits.updated.outside_proto.bytes_part1', "rb")
    updated_list1.ParseFromString(f1.read())
    f1.close()
    f2 = open('polyvore_outfits.updated.outside_proto.bytes_part2', "rb")
    updated_list2.ParseFromString(f2.read())
    f2.close()
    return updated_list1,updated_list2

def load_embeddings(item_list):
    """ map items with their color and pattern embeddings
    Args:
        item_list: item metadata
    Return:
        embedding_color_dict: python dict mapping item_ids to color embeddings
        embedding_pattern_dict: python dict mapping item_ids to pattern embeddings
    """
    embedding_color_dict = {}
    embedding_pattern_dict = {}
    image_list = read_message_from_file()
    for image in image_list.images:
        image_id = image.image_file_path.split('/')[-1][:-4]
        if (image_id not in item_list.keys()) or len(image.products)>1:
            continue
#         ground_truth_category = cate_dict[item_list[ image_id ]['category_id']]
        for product in image.products:
            if not (len(product.embedding_color) == 64 and len(product.embedding_pattern) == 64):
                continue
            embedding_color_dict[image_id] = tf.convert_to_tensor(product.embedding_color,dtype=tf.float32)
            embedding_pattern_dict[image_id] = tf.convert_to_tensor(product.embedding_pattern,dtype=tf.float32)
    return embedding_color_dict, embedding_pattern_dict

def load_embeddings_updated(item_list):
    """ map items with their updated color and pattern embeddings
    Args:
        item_list: item metadata
    Return:
        embedding_color_dict: python dict mapping item_ids to color embeddings
        embedding_pattern_dict: python dict mapping item_ids to pattern embeddings
    """
    print ('Using updated embeddings')
    embedding_color_dict = {}
    embedding_pattern_dict = {}
    updated_list1,updated_list2 = read_updated_message_from_file()
    
    # map embeddings from list 1
    for image in updated_list1.images:
        image_id = image.image_file_path.split('/')[-1][:-4]
        if (image_id not in item_list.keys()) or len(image.products)>1:
            continue
        for product in image.products:
            if not (len(product.embedding_color) == 64 and len(product.embedding_pattern) == 64):
                continue
            embedding_color_dict[image_id] = tf.convert_to_tensor(product.embedding_color,dtype=tf.float32)
            embedding_pattern_dict[image_id] = tf.convert_to_tensor(product.embedding_pattern,dtype=tf.float32)
            
    # map embeddings from list 2
    for image in updated_list2.images:
        image_id = image.image_file_path.split('/')[-1][:-4]
        if (image_id not in item_list.keys()) or len(image.products)>1:
            continue
        for product in image.products:
            if not (len(product.embedding_color) == 64 and len(product.embedding_pattern) == 64):
                continue
            embedding_color_dict[image_id] = tf.convert_to_tensor(product.embedding_color,dtype=tf.float32)
            embedding_pattern_dict[image_id] = tf.convert_to_tensor(product.embedding_pattern,dtype=tf.float32)
    print ('Done')
    return embedding_color_dict, embedding_pattern_dict

def parse_single_image(inputs):
    """ py_function to load single input image
    Args:
        inputs: [item_id, outfit_id, cate_id]
    Return:
        image: tf.float32 'Tensor' of size [im_size, im_size, 3]
        item_id: tf.int64 'Tensor'
        set_id: tf.int64 'Tensor'
        cate_id: tf.int64 'Tensor'
    """
    item_id = inputs[0]
    set_id = inputs[1]
    cate_id = inputs[2]
    image_folders = '../vasileva2018_learning/fashion-compatibility-master/data/polyvore_outfits/images'
    
    image = tf.io.read_file(os.path.join(image_folders,str(item_id.numpy())+'.jpg'))
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image,dtype=tf.float32)
    
    image /= 127.5
    image -= 1.
    
    image = tf.image.resize(image, [224, 224])
    return image,item_id,set_id,cate_id

def parse_single_embedding(inputs):
    """ py_function to load single input initial embedding
    Args:
        inputs: [item_id, outfit_id, cate_id]
    Return:
        embed: concatenated color and patten product embeddings of size [128, ]
        item_id: tf.int64 'Tensor'
        set_id: tf.int64 'Tensor'
        cate_id: tf.int64 'Tensor'
    """
    global embedding_color_dict,embedding_pattern_dict
    item_id = inputs[0]
    embed = tf.concat([embedding_color_dict[str(int(item_id.numpy()))], 
                       embedding_pattern_dict[str(int(item_id.numpy()))]], 0)
    set_id = inputs[1]
    cate_id = inputs[2]
    return embed, item_id, set_id, cate_id

def parse_triplet_images(inputs):
    """ py_function to load input images for (anchor, positive, negative) from a triplet
    Args:
        inputs: [anchor_item_id, positive_item_id, negative_item_id, anchor_cate_id, postive_cate_id]
    Return:
        anchor_image: tf.float32 'Tensor' of shape [im_size, im_size, 3]
        positive_image: tf.float32 'Tensor' of shape [im_size, im_size, 3]
        negative_image: tf.float32 'Tensor' of shape [im_size, im_size, 3]
        extra_info: tf.int64 'Tensor' with item_ids and cate_ids
    """
    image_folders = '../vasileva2018_learning/fashion-compatibility-master/data/polyvore_outfits/images'
    image_list = []
    for image_id in inputs[:3]:
        image = tf.io.read_file(os.path.join(image_folders,str(int(image_id.numpy()))+'.jpg'))
        image = tf.image.decode_jpeg(image)
        image = tf.cast(image,dtype=tf.float32)
        image /= 127.5
        image -= 1.
        image_list.append(tf.image.resize(image, [224, 224]))
    anchor_image = image_list[0]
    positive_image = image_list[1]
    negative_image = image_list[2]
    extra_info = tf.cast(inputs,tf.int64)
    anchor_cate_id = tf.cast(inputs[3],tf.int64)
    positive_cate_id = tf.cast(inputs[4],tf.int64)
    return anchor_image,positive_image,negative_image,extra_info

def parse_triplet_embeddings(inputs):
    """ py_function to load intial color and pattern embeddings for (anchor, positive, negative) from a triplet
    Args:
        inputs: [anchor_item_id, positive_item_id, negative_item_id, anchor_cate_id, postive_cate_id]
    Return:
        anchor_embed: tf.float32 'Tensor' of shape [64, ]
        positive_embed: tf.float32 'Tensor' of shape [64, ]
        negative_embed: tf.float32 'Tensor' of shape [64, ]
        extra_info: tf.int64 'Tensor' with item_ids and cate_ids
    """
    global embedding_color_dict,embedding_pattern_dict
    embed_list = []
    for image_id in inputs[:3]:
        embed_list.append(tf.concat([embedding_color_dict[str(int(image_id.numpy()))],
                                     embedding_pattern_dict[str(int(image_id.numpy()))]], 0))
    anchor_embed = embed_list[0]
    positive_embed = embed_list[1]
    negative_embed = embed_list[2]
    extra_info = tf.cast(inputs,tf.int64)
    return anchor_embed, positive_embed, negative_embed, extra_info

def reorder_item_list(mode,cate_dict,item_list):
    """ Find all possible compatible/positive pairs from all outfits
    Args:
        mode: string 'train'/'valid'/'test'
        cate_dict: python dict for semantic category mapping
        item_list: python dict for item metadata
    Return:
        paired_item_info: np array of size [2*n_pairs, 3]. 
                          Each row has item_id, outfit_id and cate_id
        item2outfit_dict: python dict mapping item ids to their outfit ids
    """
    global embedding_color_dict
    with open('../vasileva2018_learning/fashion-compatibility-master/data/polyvore_outfits/disjoint/%s.json' % mode) as f:
        outfit_ls = json.load(f)
    cate_id = {'': 0,
             'tops': 1,
             'bottoms': 2,
             'hats': 3,
             'sunglasses': 4,
             'all-body': 5,
             'outerwear': 6,
             'accessories': 7,
             'shoes': 8,
             'scarves': 9,
             'bags': 10,
             'jewellery': 11}
    pair_ls = []
    set_id_ls = []
    cate_ls = []
    item2outfit_dict = {}
    
    # enumerate all valid compatible/positive pairs
    for outfit_i in outfit_ls:
        set_id = int(outfit_i['set_id'])
        num_item = len(outfit_i['items'])
        item_ls = outfit_i['items']
        for item_i in range(num_item):
            if cate_id[ cate_dict[item_list[item_ls[item_i]['item_id']]['category_id']] ] in [0,3,4,7,9,11]:
                continue
            if item_ls[item_i]['item_id'] not in embedding_color_dict.keys():
                continue
            if int(item_ls[item_i]['item_id']) not in item2outfit_dict.keys():
                item2outfit_dict[ int(item_ls[item_i]['item_id']) ] = {set_id}
            else:
                item2outfit_dict[ int(item_ls[item_i]['item_id']) ].add(set_id)
            for item_j in range(item_i+1,num_item):
                if cate_id[ cate_dict[item_list[item_ls[item_j]['item_id']]['category_id']] ] in [0,3,4,7,9,11]:
                    continue
                if item_ls[item_j]['item_id'] not in embedding_color_dict.keys():
                    continue
                pair_ls.append([int(item_ls[item_i]['item_id']),int(item_ls[item_j]['item_id'])])
                set_id_ls.append([set_id,set_id])
                cate_ls.append([cate_id[ cate_dict[item_list[item_ls[item_i]['item_id']]['category_id']] ],
                                cate_id[ cate_dict[item_list[item_ls[item_j]['item_id']]['category_id']] ]])

    # randomly shuffle positive pairs
    rand_idx = np.arange(len(pair_ls))
    pair_ls = np.array(pair_ls)
    set_id_ls = np.array(set_id_ls)
    cate_ls = np.array(cate_ls)
    np.random.shuffle(rand_idx)
    pair_ls = pair_ls[rand_idx,:]
    set_id_ls = set_id_ls[rand_idx,:]
    cate_ls = cate_ls[rand_idx,:]
    
    # Combine item_ids, outfit_ids and cate_ids
    item_ls = [item for pair in pair_ls for item in pair]
    label_ls = [label for set_ids in set_id_ls for label in set_ids]
    type_ls = [types for cates in cate_ls for types in cates]
    paired_item_info = np.transpose(np.stack([item_ls,label_ls,type_ls]))
    return paired_item_info,item2outfit_dict


def index_all_positive_pair(pair_ls,embedding_color_dict):
    """ Build index for concatenated (anchor, positive) embeddings
    Args:
        pair_ls: item_ids for each (anchor, positive) pair
        embedding_color_dict: python dict mapping item_ids to color embeddings
    Returns:
        p_index: built index
    """
    apparel_pair_embed = []
    # get concatenated (anchor, positive) color embeddings
    for pair_info in pair_ls:
        anchor = pair_info[0]
        positive = pair_info[1]
        apparel_pair_embed.append(tf.concat([embedding_color_dict[str(int(anchor))], embedding_color_dict[str(int(positive))]],axis=0))
    apparel_pair_embed = tf.convert_to_tensor(apparel_pair_embed)
    
    # Building index with concatenated color embeddings
    dim = 128
    num_elements = apparel_pair_embed.shape[0]
    incl_pid = np.arange(num_elements)
    incl_embed = apparel_pair_embed.numpy()
    p_index= hnswlib.Index(space='cosine', dim=dim)
    p_index.init_index(max_elements=num_elements, ef_construction=100, M=16)
    p_index.add_items(incl_embed,incl_pid)
    p_index.set_ef(20)
    return p_index

def get_auxiliary_embeddings(item_ids):
    """Use initial color and pattern embeddings as auxiliary embeddings
    Args:
        item_ids: list of item ids
    Return:
        auxiliary_embeddings: tf.float32 'Tensor' of shape [n_item, embed_dim]
        auxiliary_dict: python dict mapping item_id to its auxiliary embedding
    """
    global embedding_color_dict,embedding_pattern_dict
    auxiliary_embeddings = []
    auxiliary_dict = {}
    step = 1024
    print ('Creating auxiliary embeddings...',end='')
    start_time = time.time()
    
    # concatenate color and pattern embeddings
    for i in np.arange(0,len(item_ids),step):
        embedding_ls = [tf.concat([embedding_color_dict[str(item)],
                                   embedding_pattern_dict[str(item)]],axis=0) 
                        for item in item_ids[i:min(len(item_ids),i+step)]]
        embeddings = tf.convert_to_tensor(embedding_ls)
        auxiliary_embeddings.append(embeddings)
    auxiliary_embeddings = tf.concat(auxiliary_embeddings,axis=0)
    auxiliary_embeddings = auxiliary_embeddings.numpy()
    
    # mapping item ids to auxiliary embeddings
    for i in range(auxiliary_embeddings.shape[0]):
        auxiliary_dict[item_ids[i]] = auxiliary_embeddings[i,:]
    print (time.time()-start_time,'s')
    return auxiliary_embeddings,auxiliary_dict

def build_ann_index(item_ids,auxiliary_embeddings,item_list,cate_dict):
    """ Build ANN index with auxiliary embeddings by item categories.
    Args:
        item_ids: list of item ids
        auxiliary_embeddings: tf.float32 tensor of size [n_item, embed_dim]
        item_list: item metadata
        cate_dict: python dict for semantic category mapping
    Returns:
        p: python dict containing ANN index
    """
    cate_id = {'': 0, 'tops': 1, 'bottoms': 2, 'hats': 3, 'sunglasses': 4, 'all-body': 5,
               'outerwear': 6, 'accessories': 7, 'shoes': 8, 'scarves': 9,
               'bags': 10, 'jewellery': 11}
    num_elements = auxiliary_embeddings.shape[0]
    dim  = auxiliary_embeddings.shape[1]
    cate_labels = []
    for i in range(len(item_ids)):
        cate_labels.append(cate_id[cate_dict[item_list[str(item_ids[i])]['category_id']]])
    cate_labels = np.array(cate_labels)
    p = {}
    
    # build ann index by item categories
    for target in ['tops','bottoms','bags','shoes','all-body','outerwear']:
        print ('building ANN index for',target,end=':')
        start_time = time.time()
        incl_embed = auxiliary_embeddings[cate_labels==cate_id[target],:]
        print (incl_embed.shape)
        incl_pid = item_ids[cate_labels==cate_id[target]]
        p[cate_id[target]] = hnswlib.Index(space='l2', dim=dim)
        p[cate_id[target]].init_index(max_elements=num_elements, ef_construction=100, M=16)
        p[cate_id[target]].add_items(incl_embed,incl_pid)
        print (time.time()-start_time,'s')
    return p

def retrieval_top_k(p,target_vec,k=100):
    """ Retrieval top-k nearest embeddings
    Args:
        p: ANN index dict
        target_vec: tf.float32 'Tensor' of size[embed_dim, ] for retrieval
        k: number of retrieval results returned
    Return:
        labels: list of item_ids from retrieved embeddings
    """
    p.set_ef(k)
    labels, distances=p.knn_query(target_vec,k)
    labels = labels[0]
    return labels

def triplet_item_embedding_list(args,mode,cate_dict,item_list,embedding_color_dict,embedding_pattern_dict):
    """ find all possible compatible/positive pairs from all outfits and 
        sample negatives to form triplet pairs
    Args:
        args: parsed arguments
        mode: string 'train'/'valid'/'test'
        cate_dict: python dict for semantic category mapping
        item_list: python dict for item metadata
        embedding_color_dict: python dict mapping item_ids to color embeddings
        embedding_pattern_dict: python dict mapping item_ids to pattern embeddings
    Return:
        triplet_pairs: np.array of size [2*num_anchor_positive_pair, 5]
                       each row has anchor_id, positive_id, negative_id, anchor_cate_id, positive_cate_id
    """
    with open('/home/hongruz_google_com/Documents/vasileva2018_learning/fashion-compatibility-master/data/polyvore_outfits/disjoint/%s.json' % mode) as f:
        outfit_ls = json.load(f)
    cate_id = {'': 0,
             'tops': 1,
             'bottoms': 2,
             'hats': 3,
             'sunglasses': 4,
             'all-body': 5,
             'outerwear': 6,
             'accessories': 7,
             'shoes': 8,
             'scarves': 9,
             'bags': 10,
             'jewellery': 11}
    pair_ls = []
    set_id_ls = []
    cate_ls = []
    item2outfit = {}
    cate2items = {}
    outfit2items = {}
    unique_items = set()
    # enumerate all valid (anchor, positive) pairs 
    for outfit_i in outfit_ls:
        set_id = int(outfit_i['set_id'])
        num_item = len(outfit_i['items'])
        item_ls = outfit_i['items']
        for item_i in range(num_item):
            cate_id_i = cate_id[ cate_dict[item_list[item_ls[item_i]['item_id']]['category_id']] ]
            if cate_id_i in [0,3,4,7,9,11]:
                continue
            if item_ls[item_i]['item_id'] not in embedding_color_dict.keys():
                continue
            unique_items.add(int(item_ls[item_i]['item_id']))
            if cate_id_i in cate2items.keys():
                cate2items[ cate_id_i ] = np.append(cate2items[ cate_id_i ],int(item_ls[item_i]['item_id']))
            else:
                cate2items[ cate_id_i ] = np.array(int(item_ls[item_i]['item_id']))
            if set_id in outfit2items.keys():
                outfit2items[ set_id ] = np.append(outfit2items[ set_id ],int(item_ls[item_i]['item_id']))
            else:
                outfit2items[ set_id ] = np.array(int(item_ls[item_i]['item_id']))
            if int(item_ls[item_i]['item_id']) not in item2outfit.keys():
                item2outfit[ int(item_ls[item_i]['item_id']) ] = {set_id}
            else:
                item2outfit[ int(item_ls[item_i]['item_id']) ].add(set_id)
            for item_j in range(num_item):
                cate_id_j = cate_id[ cate_dict[item_list[item_ls[item_j]['item_id']]['category_id']] ]
                if cate_id_j in [0,3,4,7,9,11]:
                    continue
                if cate_id_i == cate_id_j:
                    continue
                if item_ls[item_j]['item_id'] not in embedding_color_dict.keys():
                    continue
                pair_ls.append([int(item_ls[item_i]['item_id']),int(item_ls[item_j]['item_id'])])
                set_id_ls.append([set_id,set_id])
                cate_ls.append([cate_id_i,cate_id_j])
    cate_ls = np.array(cate_ls)
    pair_ls = np.array(pair_ls)
    set_id_ls = np.array(set_id_ls)
    
    # Start negative sampling with different choice according to:
    # args.hard_exlusion and args.hard_mining
    if mode == 'train' and args.hard_exclusion:
        # exclude false negatives and randomly sample one negative for each (anchor, positive) pair
        negative_items = np.zeros((pair_ls.shape[0],1))
        import time
        start_time = time.time()
        print ('Prepare', mode, 'hard exclusion..')
        p_index = index_all_positive_pair(pair_ls,embedding_color_dict)
        print ('Start preparing', mode, 'triplet embedding pairs...',end=' ')
        filter_cnt = 0
        for i in range(pair_ls.shape[0]):
            negative_set = np.array([])
            while negative_set.size==0:
                # randomly select a negative
                negative_set = np.random.choice(cate2items[cate_ls[i,1]],10,replace=True)
                for outfit in item2outfit[pair_ls[i,0]]:
                    negative_set = np.setdiff1d(negative_set,outfit2items[outfit])
                negative_item = np.random.choice(negative_set,1,replace=True)
                negative_pair_embed = tf.concat(
                    [embedding_color_dict[str(int(pair_ls[i,0]))], 
                     embedding_color_dict[str(int(negative_item))]],axis=0)
                negative_pair_embed = negative_pair_embed.numpy()
                
                # check if randomly selected negatives are false negatives
                labels, distances = p_index.knn_query(negative_pair_embed, k=1)
                if distances < 0.1806:
                    negative_set = np.array([])
                    filter_cnt += 1
            negative_items[i] = negative_item
        print(time.time()-start_time,'seconds')
        print('filtered:', filter_cnt,'/',pair_ls.shape[0])
        
    elif mode =='train' and args.hard_mining:
        # use initial embedding as auxiliary embeddings to select hard negatives
        
        # find unique items from the dataset
        unique_items = list(unique_items)
        unique_items = np.array(unique_items)
        
        # build ann index with auxiliary embeddings from unique items
        auxiliary_embeddings,auxiliary_dict = get_auxiliary_embeddings(unique_items)
        p = build_ann_index(unique_items,auxiliary_embeddings,item_list,cate_dict)
        
        negative_items = np.zeros((pair_ls.shape[0],1))
        import time
        start_time = time.time()
        print ('Start preparing', mode, 'triplet embedding pairs...',end=' ')
        
        # Randomly select hard negatives for a random subset of the pairs
        # If all pairs use hard negatives, the embeddings will collapse to zero
        rand_thres = 0.15 
        for i in range(pair_ls.shape[0]):
            if np.random.rand()>rand_thres:
                # Randomly sample one negative for each (anchor, positive) pair
                negative_set = np.array([])
                while negative_set.size==0:
                    negative_set = np.random.choice(cate2items[cate_ls[i,1]],10,replace=True)
                    for outfit in item2outfit[pair_ls[i,0]]:
                        negative_set = np.setdiff1d(negative_set,outfit2items[outfit])
                negative_items[i] = np.random.choice(negative_set,1,replace=True)
            else:
                # Randomly choosing one hard negative
                anchor = pair_ls[i,0]
                pos = pair_ls[i,1]
                target_vec = auxiliary_dict[anchor]
                pos_cate_id = cate_id[cate_dict[item_list[str(pos)]['category_id']]]
                search_k = 100
                negative_set = retrieval_top_k(p[pos_cate_id],target_vec,search_k)
                negative_set = negative_set[-10:]
                negative_item = np.random.choice(negative_set,1,replace=True)
                while negative_item == pos:
                    negative_item = np.random.choice(negative_set,1,replace=True)
                negative_items[i] = negative_item
        print(time.time()-start_time,'seconds')
        
    else:
        # Randomly sample one negative for each (anchor, positive) pair
        negative_items = np.zeros((pair_ls.shape[0],1))
        import time
        start_time = time.time()
        print ('Start preparing', mode, 'triplet embedding pairs...',end=' ')
        for i in range(pair_ls.shape[0]):
            negative_set = np.array([])
            while negative_set.size==0:
                # randomly select a negative
                negative_set = np.random.choice(cate2items[cate_ls[i,1]],10,replace=True)
                for outfit in item2outfit[pair_ls[i,0]]:
                    negative_set = np.setdiff1d(negative_set,outfit2items[outfit])
            negative_items[i] = np.random.choice(negative_set,1,replace=True)
        print(time.time()-start_time,'seconds')
    
    triplet_pairs = np.concatenate((pair_ls,negative_items,cate_ls),axis=1)
    np.random.shuffle(triplet_pairs)
    return triplet_pairs

def get_batch_dataset(mode,args):
    """ Return batched tf dataset
    Args:
        mode: 'train'/ 'valid'/ 'test'
        args: parsed arguments
    """
    global embedding_color_dict,embedding_pattern_dict,item2outfit_dict
    cate_dict,item_list = read_files()
    # load color and pattern embeddings. will also exclude items not on the whitelist
    if args.use_updated_embedding:
        embedding_color_dict,embedding_pattern_dict = load_embeddings_updated(item_list)
    else:
        embedding_color_dict,embedding_pattern_dict = load_embeddings(item_list)
    item2outfit_dict = None
    if mode == 'train' and not args.triplet_input: # Select multiple negative for each (anchor, positive) pair
        item_ds,item2outfit_dict = reorder_item_list(mode,cate_dict,item_list)
        print ('total # of',mode,'positive pairs:', item_ds.shape[0])
        list_ds = tf.data.Dataset.from_tensor_slices(item_ds)
        if args.embed_input:
            print ('Using embed input')
            embedding_ds = list_ds.map(lambda x: tf.py_function(parse_single_embedding, 
                                                                [x], 
                                                                [tf.float32,tf.int64,tf.int64, tf.int64]))
            batched_dataset = embedding_ds.batch(args.batch_size)
        else:
            print ('Using image input')
            images_ds = list_ds.map(lambda x: tf.py_function(parse_single_image, 
                                                                 [x],
                                                                 [tf.float32, tf.int64, tf.int64, tf.int64]))
            batched_dataset = images_ds.batch(args.batch_size)
    else: # Select one negative for each (anchor, positive) pair
        item_ds = triplet_item_embedding_list(args,mode,cate_dict,item_list,embedding_color_dict,embedding_pattern_dict)
        print ('total # of',mode,'triplet pairs:', item_ds.shape[0])
        list_ds = tf.data.Dataset.from_tensor_slices(item_ds)
        if args.embed_input:
            print ('Using embed input')
            embedding_ds = list_ds.map(lambda x: tf.py_function(parse_triplet_embeddings, 
                                                                [x], 
                                                                [tf.float32,tf.float32,tf.float32, tf.int64]))
            batched_dataset = embedding_ds.batch(args.batch_size)
        else:
            print ('Using image input')
            images_ds = list_ds.map(lambda x: tf.py_function(parse_triplet_images, 
                                                             [x],
                                                             [tf.float32,tf.float32,tf.float32,tf.int64]))
            batched_dataset = images_ds.batch(args.batch_size)
    return batched_dataset,item2outfit_dict

def load_fitb_questions(embedding_color_dict,cate_dict,item_list,cate_id):
    """ Load fill-in-the-blank test questions
    Args:
        embedding_color_dict: python dict mapping item_ids to color embeddings
        cate_dict: python dict for semantic category mapping
        item_list: item metadata
        cate_id: mapping from category_names to category_ids
    Return:
        q_list: list of fitb questions
        a_list: list of fitb answers
        gt_list: list of ground-truth answers
    """
    with open('../vasileva2018_learning/fashion-compatibility-master/data/polyvore_outfits/disjoint/test.json') as f:
        outfit_ls = json.load(f)
    idx2item_id = {}
    for outfit in outfit_ls:
        outfit_id = int(outfit['set_id'])
        idx2item_id[outfit_id] = {}
        for each in outfit['items']:
            idx2item_id[outfit_id][each['index']] = int(each['item_id'])
    fitbdir = '../vasileva2018_learning/fashion-compatibility-master/data/polyvore_outfits/disjoint/fill_in_blank_test.json'
    with open(fitbdir) as f:
        fitb_questions = json.load(f)
    q_list = []
    a_list = []
    gt_list = []
    
    # iterate through all fitb_questions and remove invalid products
    for line in fitb_questions:
        q_items = []
        incl_tmp = []
        tmp = []
        incl = True
        for token in line['question']:
            subtoken = token.split('_')
            item_id = idx2item_id[int(subtoken[0])][int(subtoken[1])]
            gt = int(subtoken[0])
            cate_i = cate_id[ cate_dict[item_list[str(item_id)]['category_id']] ]
            if cate_i in [0,3,4,7,9,11] or str(item_id) not in embedding_color_dict.keys():
                continue
            q_items.append(item_id)
            tmp.append(token)
        if len(q_items) < 1:
            incl = False
        valid_a = True
        set_id_ls = []
        a_items = []
        for token in line['answers']:
            subtoken = token.split('_')
            item_id = idx2item_id[int(subtoken[0])][int(subtoken[1])]
            cate_i = cate_id[ cate_dict[item_list[str(item_id)]['category_id']] ]
            if cate_i in [0,3,4,7,9,11] or str(item_id) not in embedding_color_dict.keys():
                valid_a = False
                break
            a_items.append(item_id)
            set_id_ls.append(int(subtoken[0]))

        incl = valid_a and incl
        if incl:
            q_list.append(q_items)
            a_list.append(a_items)
            gt_list.append(np.asarray(gt==np.array(set_id_ls)).nonzero())
    return q_list,a_list,gt_list

def fitb_test(args,model):
    """ Test model performance on Fill-in-the-blank task
    Args:
        model: embedding model
    Return:
        fitb_acc: Fill-in-the-blank accuracy
    """
    cate_dict,item_list = read_files()
    cate_id = {'': 0,'tops': 1,'bottoms': 2,'hats': 3,'sunglasses': 4,'all-body': 5,'outerwear': 6,
             'accessories': 7,'shoes': 8,'scarves': 9,'bags': 10,'jewellery': 11}
    if args.use_updated_embedding:
        embedding_color_dict,embedding_pattern_dict = load_embeddings_updated(item_list)
    else:
        embedding_color_dict,embedding_pattern_dict = load_embeddings(item_list)
    q_list,a_list,gt_list = load_fitb_questions(embedding_color_dict,cate_dict,item_list,cate_id)
    
    # record unique product from q_list and a_list
    unique_prod = set()
    for q in q_list:
        for each in q:
            unique_prod.add(each)
    for a in a_list:
        for each in a:
            unique_prod.add(each)
    unique_prod = list(unique_prod)
    unique_cate = [cate_id[ cate_dict[item_list[str(item_id)]['category_id']] ] for item_id in unique_prod]
    
    # calcuate compatible embeddings for each unique product
    step = 1024
    model_embeddings = []
    for i in np.arange(0,len(unique_prod),step):
        embedding_ls = [tf.concat([embedding_color_dict[str(item)],
                                   embedding_pattern_dict[str(item)]],axis=0) 
                        for item in unique_prod[i:min(len(unique_prod),i+step)]]
        embeddings = tf.convert_to_tensor(embedding_ls)
        label_ls = [cate for cate in unique_cate[i:min(len(unique_cate),i+step)]]
        label_ls = tf.convert_to_tensor(label_ls)
        label_onehot = tf.one_hot(label_ls,12)
        embeddings = model(embeddings,label_onehot)
        model_embeddings.append(embeddings)
    model_embeddings = tf.concat(model_embeddings,axis=0)
    embedding_dict = {}
    for i in range(model_embeddings.shape[0]):
        embedding_dict[unique_prod[i]] = model_embeddings[i,:]
        
    # calculate fill-in-the-blank accuracy
    acc = []
    for i in range(len(q_list)):
        embedding_ls =[embedding_dict[item] for item in q_list[i]]
        q_embeddings = tf.convert_to_tensor(embedding_ls)
        embedding_ls =[embedding_dict[item] for item in a_list[i]]
        a_embeddings = tf.convert_to_tensor(embedding_ls)
        all_embeddings = tf.concat([q_embeddings,a_embeddings],axis=0)
        pairwise_dist = tfa.losses.metric_learning.pairwise_distance(all_embeddings)
        dist_mat = pairwise_dist[q_embeddings.shape[0]:,:q_embeddings.shape[0]]
        avg_dist = tf.reduce_mean(dist_mat,axis=1)
        is_correct = tf.argmin(avg_dist)==gt_list[i]
        acc.append(is_correct)
    fitb_acc = np.sum(acc)/len(q_list)
    return fitb_acc

def load_comp_prediction_questions(embedding_color_dict,cate_dict,item_list,cate_id):
    """ Load compatibility prediction questions
    Args:
        embedding_color_dict: python dict mapping item_ids to color embeddings
        cate_dict: python dict for semantic category mapping
        item_list: item metadata
        cate_id: mapping from category_names to category_ids
    Return:
        q_list: list of compatibility prediction questions
        a_list: list of compatibility prediction answers [0/1]
    """
    compdir = '../vasileva2018_learning/fashion-compatibility-master/data/polyvore_outfits/disjoint/compatibility_test.txt'
    f = open(compdir)
    questions = f.read().split('\n')
    with open('../vasileva2018_learning/fashion-compatibility-master/data/polyvore_outfits/disjoint/test.json') as f:
        outfit_ls = json.load(f)
    idx2item_id = {}
    for outfit in outfit_ls:
        outfit_id = int(outfit['set_id'])
        idx2item_id[outfit_id] = {}
        for each in outfit['items']:
            idx2item_id[outfit_id][each['index']] = int(each['item_id'])
    
    # iterate through all fitb_questions and remove invalid products
    q_list = []
    a_list = []
    for question in questions:
        token = question.split(' ')
        ans= int(token[0])
        incl = True
        item_ls = []
        tmp=[]
        for i in range (1,len(token)):
            subtoken = token[i].split('_')
            if len(subtoken)==2:
                item_id = idx2item_id[int(subtoken[0])][int(subtoken[1])]
                cate_i = cate_id[ cate_dict[item_list[str(item_id)]['category_id']] ]
                if cate_i in [0,3,4,7,9,11] or str(item_id) not in embedding_color_dict.keys():
                    continue
                item_ls.append(item_id)
        if len(item_ls) <2:
            incl = False
        if incl:
            q_list.append(item_ls)
            a_list.append(ans)
    return q_list, a_list

def comp_prediction_test(args,model):
    """ Test model performance on compatiblity prediction task
    Args:
        args: parsed arguments
        model: embedding model
    Return:
        comp_roc_auc_score: compatiblity prediction AUC
    """
    cate_dict,item_list = read_files()
    cate_id = {'': 0,'tops': 1,'bottoms': 2,'hats': 3,'sunglasses': 4,'all-body': 5,'outerwear': 6,
             'accessories': 7,'shoes': 8,'scarves': 9,'bags': 10,'jewellery': 11}
    if args.use_updated_embedding:
        embedding_color_dict,embedding_pattern_dict = load_embeddings_updated(item_list)
    else:
        embedding_color_dict,embedding_pattern_dict = load_embeddings(item_list)
    q_list,a_list = load_comp_prediction_questions(embedding_color_dict,cate_dict,item_list,cate_id)
    # record unique product from q_list
    unique_prod = set()
    for q in q_list:
        for each in q:
            unique_prod.add(each)
    unique_prod = list(unique_prod)
    unique_cate = [cate_id[ cate_dict[item_list[str(item_id)]['category_id']] ] for item_id in unique_prod]
    print (len(unique_prod))
    
    # calcuate compatible embeddings for each unique product
    step = 1024
    model_embeddings = []
    for i in np.arange(0,len(unique_prod),step):
        embedding_ls = [tf.concat([embedding_color_dict[str(item)],
                                   embedding_pattern_dict[str(item)]],axis=0) 
                        for item in unique_prod[i:min(len(unique_prod),i+step)]]
        embeddings = tf.convert_to_tensor(embedding_ls)
        label_ls = [cate for cate in unique_cate[i:min(len(unique_cate),i+step)]]
        label_ls = tf.convert_to_tensor(label_ls)
        label_onehot = tf.one_hot(label_ls,12)
        embeddings = model(embeddings,label_onehot)
        model_embeddings.append(embeddings)
    model_embeddings = tf.concat(model_embeddings,axis=0)
    embedding_dict = {}
    for i in range(model_embeddings.shape[0]):
        embedding_dict[unique_prod[i]] = model_embeddings[i,:]
    
    # calculate compatiblity score for each outfit
    score_ls = []
    for i in range(len(q_list)):
        embedding_ls =[embedding_dict[item] for item in q_list[i]]
        embeddings = tf.convert_to_tensor(embedding_ls)
        pairwise_dist = tfa.losses.metric_learning.pairwise_distance(embeddings)
        mask = tf.cast(tf.eye(pairwise_dist.shape[0])==0,dtype=tf.float32)
        score = tf.reduce_sum(tf.multiply(pairwise_dist,mask)).numpy()/(mask.shape[0]*(mask.shape[0]-1))
        score_ls.append(score)
    
    comp_roc_auc_score = roc_auc_score(1-np.array(a_list),np.array(score_ls))
    return comp_roc_auc_score