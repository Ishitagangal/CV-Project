# -*- coding: utf-8 -*-
def adjust_learning_rate(optimizer, epoch, init_lr=0.1, step=30, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (decay ** (epoch // step))
    print('Learning Rate %f'%lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    #print("maxk {} batchsize {} output {} target {} ".format(maxk,batch_size,output,target)) 
    someval, pred = output.topk(maxk, 1, True, True)
    #print("someval {} pred {} ".format(someval,pred))
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #print(" correct {} ".format(correct))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
        #print("correct_k {} for k {}".format(correct_k,k))
    return res

