from sklearn.metrics import average_precision_score
import numpy as np
import torch

criterion = torch.nn.CrossEntropyLoss(reduction='mean')

y_tru = torch.tensor([5])
x_true = np.array([[1,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,1]])
y_true = np.array([[0,0,0,0,0,1,0,0,0,0,0]])
# print(x_true.shape)
# x_true = x_true.reshape(1,11)
# print(x_true)
# print(x_true.shape)

x_scores = np.array([-0.0503,  0.0406,  0.0755,  0.1086, -0.0618, -0.0631, -0.2993, -0.0532,-0.0461, -0.2780,  0.4034])
# print(x_scores.shape)

soft_x_scores = np.array([[0.08674151, 0.09499384, 0.0983683 , 0.10168085, 0.08574722, 0.08563477, 0.06762342, 0.08649036, 0.08710556, 0.06907924, 0.13653497],[0.08674151, 0.09499384, 0.0983683 , 0.10168085, 0.08574722, 0.08563477, 0.06762342, 0.08649036, 0.08710556, 0.06907924, 0.13653497],[0.08674151, 0.09499384, 0.0983683 , 0.10168085, 0.08574722, 0.08563477, 0.06762342, 0.08649036, 0.08710556, 0.06907924, 0.13653497],[0.08674151, 0.09499384, 0.0983683 , 0.10168085, 0.08574722, 0.08563477, 0.06762342, 0.08649036, 0.08710556, 0.06907924, 0.13653497],[0.08674151, 0.09499384, 0.0983683 , 0.10168085, 0.08574722, 0.08563477, 0.06762342, 0.08649036, 0.08710556, 0.06907924, 0.13653497],[0.08674151, 0.09499384, 0.0983683 , 0.10168085, 0.08574722, 0.08563477, 0.06762342, 0.08649036, 0.08710556, 0.06907924, 0.13653497],[0.08674151, 0.09499384, 0.0983683 , 0.10168085, 0.08574722, 0.08563477, 0.06762342, 0.08649036, 0.08710556, 0.06907924, 0.13653497],[0.08674151, 0.09499384, 0.0983683 , 0.10168085, 0.08574722, 0.08563477, 0.06762342, 0.08649036, 0.08710556, 0.06907924, 0.13653497],[0.08674151, 0.09499384, 0.0983683 , 0.10168085, 0.08574722, 0.08563477, 0.06762342, 0.08649036, 0.08710556, 0.06907924, 0.13653497],[0.08674151, 0.09499384, 0.0983683 , 0.10168085, 0.08574722, 0.08563477, 0.06762342, 0.08649036, 0.08710556, 0.06907924, 0.13653497],[0.08674151, 0.09499384, 0.0983683 , 0.10168085, 0.08574722, 0.08563477, 0.06762342, 0.08649036, 0.08710556, 0.06907924, 0.13653497]])
print(soft_x_scores.shape)

sigma_x_scores = np.array([[0.4874, 0.5101, 0.5189, 0.5271, 0.4846, 0.4842, 0.4257, 0.4867, 0.4885, 0.4310, 0.5995]])
# print(sigma_x_scores.shape)
# sigma_x_scores = sigma_x_scores.reshape(1,11)
# print(sigma_x_scores)
# print(sigma_x_scores.shape)

# aps  = average_precision_score(x_true,soft_x_scores,average="macro")
# print(aps)

x = sigma_x_scores[0]
print(x)

for i, value in enumerate(x):
    if (value>0.5):
        x[i] = 1
    else:
        x[i] = 0

x = x.reshape(1,11)

def get_ap_score(y_true, y_scores):
    """
    Get average precision score between 2 1-d numpy arrays
    
    Args:
        y_true: batch of true labels
        y_scores: batch of confidence scores
=
    Returns:
        sum of batch average precision
    """
    #pdb.set_trace()
    scores = 0.0
    #y_true = y_true.reshape(1,1)
    for i in range(y_true.shape[0]):
        out = average_precision_score(y_true = y_true[i], y_score = y_scores[i],average=None)
        scores += out
        print(out)
    
    return scores


# out = get_ap_score(y_true,sigma_x_scores)
# print(out)

# out = average_precision_score(y_true,x)
# print(out)
# print(y_true)
# print(x)
y_tru = torch.tensor([5])
x_scores = np.array([-0.0503,  -0.0406,  -0.0755,  -0.1086, -0.0618, 0.4034, -0.2993, -0.0532,-0.0461, -0.2780, -0.0631])
x_scores = torch.from_numpy(x_scores)
x_scores = x_scores.reshape(1,11)
print(x_scores.shape)
loss = criterion(x_scores,y_tru)
print(loss)