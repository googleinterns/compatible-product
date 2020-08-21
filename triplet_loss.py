"""Define functions to create the triplet loss with online triplet mining."""

import tensorflow as tf


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0),dtype=tf.float32)
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

def _masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.
    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
    masked_maximums = (
        tf.math.reduce_max(
            tf.math.multiply(data - axis_minimums, mask), dim, keepdims=True
        )
        + axis_minimums
    )
    return masked_maximums


def _masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.
    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
    masked_minimums = (
        tf.math.reduce_min(
            tf.math.multiply(data - axis_maximums, mask), dim, keepdims=True
        )
        + axis_maximums
    )
    return masked_minimums

def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask

def _get_anchor_positive_triplet_mask_2D(labels_mask):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels_mask: labels of 2d label masks, of size [batch_size, batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(labels_mask.shape[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels_mask
    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask_2D(labels_mask,pos_cates,cate_ids):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct set_id labels and same category labels.
    Args:
        labels_mask: labels of 2d label masks, of size [batch_size, batch_size]
        pos_cates: category labels for positives, tf.int32 `Tensor` with shape [batch_size, ]
        cate_ids: category labels, tf.int32 `Tensor` with shape [batch_size, ]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels_mask
    cate_equal = tf.equal(tf.expand_dims(pos_cates,1),tf.expand_dims(cate_ids,0))
    mask = tf.logical_and(tf.logical_not(labels_equal),cate_equal)

    return mask

def _get_category_mask(labels):
    """Return a 3D mask where mask[a,p,n] in True iff a, p, n conform to:
        category(a) != category(p) &
        category(p) == category(n)
    Args:
        labels: tf.int32 'Tensor' with shape [batch_size]
    Returns:
        cate_mask: tf.bool 'Tensor' with shape [batch_size, batch_size, batch_size]
    """
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    j_equal_k = tf.expand_dims(label_equal, 0)
    cate_mask = tf.logical_and(tf.logical_not(i_equal_j), j_equal_k)
    return cate_mask

def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool 'Tensor' with shape [batch_size, batch_size, batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask

def _get_triplet_mask_from2D(labels_mask):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels_mask: labels of 2d label masks, of size [batch_size, batch_size]
    Return:
        mask: tf.bool 'Tensor' with shape [batch_size, batch_size, batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(labels_mask.shape[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = labels_mask
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask

def batch_all_triplet_loss(labels, cate_ids, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size [batch_size,] aka set_ids
        cate_ids: category labels from the batch of size [batch_size,]
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
        fraction_positive_triplets: scalar tensor containing the frction of valid positive triplets
        mean_dist: scalar tensor containing the mean of all pairwise distances
        mask: tf.float32 'Tensor' with shape [batch_size, batch_size, batch_size]
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    
    # Get masks for category labels
    cate_mask = _get_category_mask(cate_ids)
    
    #Combine masks for valid triplets and masks for categories
    mask = tf.logical_and(mask,cate_mask)
    
    mask = tf.cast(mask,dtype=tf.float32)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16),dtype=tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
    mean_dist = tf.reduce_mean(pairwise_dist)
    return triplet_loss, fraction_positive_triplets, mean_dist, mask

def batch_all_triplet_loss_v2(labels_mask, cate_ids, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels_mask: labels of 2d label masks, of size [batch_size, batch_size] aka set_ids
        cate_ids: category labels from the batch of size [batch_size, ]
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
        fraction_positive_triplets: scalar tensor containing the frction of valid positive triplets
        mean_dist: scalar tensor containing the mean of all pairwise distances
        mask: tf.float32 'Tensor' with shape [batch_size, batch_size, batch_size]
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask_from2D(labels_mask)
    
    # Get masks for category labels
    cate_mask = _get_category_mask(cate_ids)
    
    #Combine masks for valid triplets and masks for categories
    mask = tf.logical_and(mask,cate_mask)
    
    mask = tf.cast(mask,dtype=tf.float32)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16),dtype=tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_valid_triplets + 1e-16)
    mean_dist = tf.reduce_mean(pairwise_dist)
    return triplet_loss, fraction_positive_triplets, mean_dist, mask

def triplet_loss_by_pairs(dist_ap,dist_an,margin):
    """ Build the average triplet loss over a batch of (a, p, n) pairs
    Args:
        dist_ap: distances between anchors and positives, of size [batch_size, ]
        dist_an: distances between anchors and negatives, of size [batch_size, ]
        margin: margin for triplet loss
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
        fraction_positive_triplets: scalar tensor containing the frction of valid positive triplets
        mean_dist: scalar tensor containing the mean of all pairwise distances
    """
    mean_dist = tf.reduce_mean(tf.concat([dist_ap,dist_an],0))
    triplet_loss = dist_ap - dist_an + margin
    triplet_loss = tf.maximum(triplet_loss, 0.0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16),dtype=tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    triplet_loss = tf.reduce_sum(triplet_loss) /dist_ap.shape[0] #(num_positive_triplets + 1e-16)
    fraction_positive_triplets = num_positive_triplets / (dist_ap.shape[0] + 1e-16)
    return triplet_loss,fraction_positive_triplets,mean_dist



def batch_hard_triplet_loss_v2(labels_mask, cate_ids, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels_mask: labels of 2d label masks, of size [batch_size, batch_size]
        embeddings: tensor of shape [batch_size, embed_dim]
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
        positive_fractions: scalar tensor containing the frction of valid positive triplets
        mean_dist: scalar tensor containing the mean of all pairwise distances
        mask_anchor_negative: tf.float32 'Tensor' with shape [batch_size, batch_size]
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask_2D(labels_mask)
    mask_anchor_positive = tf.cast(mask_anchor_positive,dtype=tf.float32)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    pos_cates = tf.nn.embedding_lookup(cate_ids,tf.argmax(anchor_positive_dist, axis=1))
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask_2D(labels_mask,pos_cates,cate_ids)
    mask_anchor_negative = tf.cast(mask_anchor_negative,dtype=tf.float32)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    valid_triplet_mask = tf.greater(tf.reduce_sum(mask_anchor_negative,axis=1),0.0)
    positive_triplet_mask = tf.logical_and(tf.greater(tf.reshape(triplet_loss,(triplet_loss.shape[0],)),0.0), valid_triplet_mask)
    num_positive_triplets = tf.reduce_sum(tf.cast( positive_triplet_mask,dtype=tf.float32))
    num_valid_triplets = tf.reduce_sum(tf.cast(tf.reduce_sum(mask_anchor_negative,axis=1)>0,dtype=tf.float32))
    positive_fractions = 1.0 * num_positive_triplets / (num_valid_triplets+1e-16)
    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss[tf.reduce_sum(mask_anchor_negative,axis=1)>0])
    mean_dist = tf.reduce_mean(pairwise_dist)
    return triplet_loss,positive_fractions,mean_dist,mask_anchor_negative



def batch_semihard_triplet_loss_v2(labels_mask, cate_ids, embeddings, margin, squared=False):
    """Build the semihard triplet loss over a batch of embeddings.
    For each anchor, we get the positive and semihard negative to form a triplet.
    Args:
        labels_mask: labels of 2d label masks, of size [batch_size, batch_size]
        embeddings: tensor of shape [batch_size, embed_dim]
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
        positive_fractions: scalar tensor containing the frction of valid positive triplets
        mean_dist: scalar tensor containing the mean of all pairwise distances
        mask_anchor_negative: tf.float32 'Tensor' with shape [batch_size, batch_size]
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask_2D(labels_mask)
    mask_anchor_positive = tf.cast(mask_anchor_positive,dtype=tf.float32)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    pos_cates = tf.nn.embedding_lookup(cate_ids,tf.argmax(anchor_positive_dist, axis=1))
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask_2D(labels_mask,pos_cates,cate_ids)
    mask_anchor_negative = tf.cast(mask_anchor_negative,dtype=tf.float32)
    
    # Get a mask to indicate where anchor-negative distance is greater or smaller than anchor-positive distance
    mask_negative_greater = tf.greater(pairwise_dist,hardest_positive_dist)
    mask_negative_greater = tf.cast(mask_negative_greater,dtype=tf.float32)
    mask_negative_smaller = tf.multiply(1.0-mask_negative_greater, mask_anchor_negative)
    mask_negative_greater = tf.multiply(mask_negative_greater,mask_anchor_negative)
    
    # Get negatives with d(a,n) > d(a,p) and negatives with d(a,n)<=d(a,p)
    negative_greater = _masked_minimum(pairwise_dist, mask_negative_greater, dim=1)
    negative_smaller = _masked_maximum(pairwise_dist, mask_negative_smaller, dim=1)
    mask_exist_negative_greater = tf.math.greater(
        tf.math.reduce_sum(mask_negative_greater, 1, keepdims=True),0.0)
    mask_exist_negative_smaller = tf.math.greater(
        tf.math.reduce_sum(mask_negative_smaller, 1, keepdims=True),0.0)
    mask_valid_triplet = tf.math.logical_or(mask_exist_negative_greater,mask_exist_negative_smaller)
    
    # Semihard negatives: if exists n such that d(a,n)>d(a,p), then select the n from smallest d(a,n). 
    # Otherwise select n for the largest d(a,n)
    semihard_negative_dist = tf.where(mask_exist_negative_greater, negative_greater, negative_smaller)
    triplet_loss = tf.maximum(hardest_positive_dist - semihard_negative_dist + margin, 0.0)
    triplet_loss = triplet_loss[mask_valid_triplet]
    positive_triplet_mask = tf.greater(triplet_loss, 0.0)
    num_positive_triplets = tf.reduce_sum(tf.cast(positive_triplet_mask,dtype=tf.float32))
    num_valid_triplets = tf.reduce_sum(tf.cast(mask_valid_triplet,dtype=tf.float32))
    positive_fractions = 1.0 * num_positive_triplets / (num_valid_triplets+1e-16)
    triplet_loss = tf.reduce_mean(triplet_loss)
    mean_dist = tf.reduce_mean(pairwise_dist)
    return triplet_loss,positive_fractions,mean_dist,mask_anchor_negative

def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
        mean_dist: scalar tensor containing the mean of all pairwise distances
        mean_hard_negative_dist: scalar tensor containing the mean of all hard negatives' distances
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.cast(mask_anchor_positive,dtype=tf.float32)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.cast(mask_anchor_negative,dtype=tf.float32)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)
    mean_dist = tf.reduce_mean(hardest_positive_dist)
    mean_hard_negative_dist = tf.reduce_mean(hardest_negative_dist)
    return triplet_loss,mean_dist,mean_hard_negative_dist