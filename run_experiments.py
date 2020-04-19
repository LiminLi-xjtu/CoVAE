## This program is based on the DeepDTA model(https://github.com/hkmztrk/DeepDTA)
## The program requires pytorch and gpu support.
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import os
from tqdm import tqdm
from math import exp
os.environ['PYTHONHASHSEED'] = '0'
import matplotlib

matplotlib.use('Agg')

from datahelper import *
from arguments import argparser, logging
import time
import matplotlib.pyplot as plt
from copy import deepcopy
from emetrics import *
from model import net
from sklearn.metrics import roc_auc_score, accuracy_score

figdir = "figures/"


def get_random_folds(tsize, foldcount):
    folds = []
    indices = set(range(tsize))
    foldsize = tsize / foldcount
    leftover = tsize % foldcount
    for i in range(foldcount):
        sample_size = foldsize
        if leftover > 0:
            sample_size += 1
            leftover -= 1
        fold = random.sample(indices, int(sample_size))
        indices = indices.difference(fold)
        folds.append(fold)

    # assert stuff
    foldunion = set([])
    for find in range(len(folds)):
        fold = set(folds[find])
        assert len(fold & foldunion) == 0, str(find)
        foldunion = foldunion | fold
    assert len(foldunion & set(range(tsize))) == tsize

    return folds


def get_drugwise_folds(label_row_inds, label_col_inds, drugcount, foldcount):
    assert len(np.array(label_row_inds).shape) == 1, 'label_row_inds should be one dimensional array'
    row_to_indlist = {}
    rows = sorted(list(set(label_row_inds)))
    for rind in rows:
        alloccs = np.where(np.array(label_row_inds) == rind)[0]
        row_to_indlist[rind] = alloccs
    drugfolds = get_random_folds(drugcount, foldcount)
    folds = []
    for foldind in range(foldcount):
        fold = []
        drugfold = drugfolds[foldind]
        for drugind in drugfold:
            fold = fold + row_to_indlist[drugind].tolist()
        folds.append(fold)
    return folds


def get_targetwise_folds(label_row_inds, label_col_inds, targetcount, foldcount):
    assert len(np.array(label_col_inds).shape) == 1, 'label_col_inds should be one dimensional array'
    col_to_indlist = {}
    cols = sorted(list(set(label_col_inds)))
    for cind in cols:
        alloccs = np.where(np.array(label_col_inds) == cind)[0]
        col_to_indlist[cind] = alloccs
    target_ind_folds = get_random_folds(targetcount, foldcount)
    folds = []
    for foldind in range(foldcount):
        fold = []
        targetfold = target_ind_folds[foldind]
        for targetind in targetfold:
            fold = fold + col_to_indlist[targetind].tolist()
        folds.append(fold)
    return folds


def loss_f(recon_x, x, mu, logvar):

    cit = nn.CrossEntropyLoss(reduction='none')
    cr_loss = torch.sum(cit(recon_x.permute(0, 2, 1), x), 1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
    return torch.mean(cr_loss + KLD)


def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
        if isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight.data, 1)
            init.constant_(m.bias.data, 0)
        if isinstance(m, nn.LSTM):
            init.orthogonal_(m.all_weights[0][0])
            init.orthogonal_(m.all_weights[0][1])
        if isinstance(m, nn.Conv1d):
            init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0)


def train(train_loader, model, FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2,lamda):
    model.train()
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    with tqdm(train_loader) as t:
        for drug_SMILES, target_protein, affinity in t:
            drug_SMILES = torch.Tensor(drug_SMILES)
            target_protein = torch.Tensor(target_protein)
            affinity = torch.Tensor(affinity)
            optimizer.zero_grad()

            affinity = Variable(affinity).cuda()
            pre_affinity, new_drug, new_target, drug, target, mu_drug, logvar_drug, mu_target, logvar_target = model(
                    drug_SMILES, target_protein, FLAGS, NUM_FILTERS,
                    FILTER_LENGTH1, FILTER_LENGTH2)
            loss_affinity = loss_func(pre_affinity, affinity)
            loss_drug = loss_f(new_drug, drug, mu_drug, logvar_drug)
            loss_target = loss_f(new_target, target, mu_target, logvar_target)
            c_index = get_cindex(affinity.cpu().detach().numpy(),
                                     pre_affinity.cpu().detach().numpy())
            loss = loss_affinity + 10**lamda * (loss_drug + FLAGS.max_smi_len / FLAGS.max_seq_len * loss_target)
            loss.backward()
            optimizer.step()
            mse = loss_affinity.item()
            t.set_postfix(train_loss=loss.item(), mse=mse, train_cindex=c_index)
    return model

def test(model,test_loader,FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2, lamda):
    model.eval()
    loss_func = nn.MSELoss()
    affinities = []
    pre_affinities = []
    loss_d=0
    loss_t=0
    with torch.no_grad():
        for i,(drug_SMILES, target_protein, affinity) in enumerate(test_loader):
            pre_affinity, new_drug, new_target, drug, target, mu_drug, logvar_drug, mu_target, logvar_target = model(drug_SMILES, target_protein, FLAGS, NUM_FILTERS,
                                                        FILTER_LENGTH1, FILTER_LENGTH2)
            pre_affinities += pre_affinity.cpu().detach().numpy().tolist()
            affinities += affinity.cpu().detach().numpy().tolist()
            loss_d+=loss_f(new_drug, drug, mu_drug, logvar_drug)
            loss_t+=loss_f(new_target, target, mu_target, logvar_target)
        pre_affinities = np.array(pre_affinities)
        affinities = np.array(affinities)
        loss = loss_func(torch.Tensor(pre_affinities), torch.Tensor(affinities))
        cindex = get_cindex(affinities,pre_affinities)
        rm2 = get_rm2(affinities, pre_affinities)
        auc = roc_auc_score(np.int32(affinities >7), pre_affinities)
    return cindex,loss,rm2,auc

def nfold_setting_sample(XD, XT, Y, label_row_inds, label_col_inds, measure, FLAGS, dataset, nfolds,i):
    test_set = nfolds[5]
    outer_train_sets = nfolds[0:5]
    # test_set, outer_train_sets=dataset.read_sets(FLAGS)

    # if FLAGS.problem_type==1:
    #     test_set, outer_train_sets = dataset.read_sets(FLAGS)
    foldinds = len(outer_train_sets)
    ## TRAIN AND VAL
    val_sets = []
    train_sets = []
    test_sets = []
    for val_foldind in range(foldinds):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        train_sets.append(otherfoldsinds)
        test_sets.append(test_set)
        print("val set", str(len(val_sets)))
        print("train set", str(len(train_sets)))

    bestparamind, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv(XD, XT, Y, label_row_inds,
                                                                                            label_col_inds, measure,
                                                                                            FLAGS, train_sets,
                                                                                            test_sets, get_aupr,
                                                                                            get_rm2)

    best_param, bestperf, all_predictions, all_losses, all_auc, all_aupr = general_nfold_cv_test(XD, XT, Y, label_row_inds, label_col_inds,
                                                                              measure, FLAGS, train_sets, test_sets,
                                                                              get_rm2, best_param_list,i)

    logging("---FINAL RESULTS-----", FLAGS)
    logging("best param = %s" % best_param_list, FLAGS)

    testperfs = []
    testloss = []
    testauc = []
    testaupr = []
    avgperf = 0.

    for test_foldind in range(len(test_sets)):
        foldperf = all_predictions[test_foldind]
        foldloss = all_losses[test_foldind]
        testperfs.append(foldperf)
        testloss.append(foldloss)
        testauc.append(all_auc[test_foldind])
        testaupr.append(all_aupr[test_foldind])
        avgperf += foldperf

    avgperf = avgperf / len(test_sets)
    avgloss = np.mean(testloss)
    perf_std = np.std(testperfs)
    loss_std = np.std(testloss)
    avg_auc = np.mean(testauc)
    auc_std = np.std(testauc)
    avg_aupr = np.mean(testaupr)
    aupr_std = np.std(testaupr)

    logging("Test Performance CI", FLAGS)
    logging(testperfs, FLAGS)
    logging("Test Performance MSE", FLAGS)
    logging(testloss, FLAGS)

    print(best_param_list)
    print('averaged performance', avgperf)
    return avgperf, avgloss, perf_std, loss_std, avg_auc, auc_std, avg_aupr, aupr_std


def general_nfold_cv(XD, XT, Y, label_row_inds, label_col_inds, prfmeasure, FLAGS, labeled_sets, val_sets, get_aupr,
                     get_rm2):  

    paramset1 = FLAGS.num_windows 
    paramset2 = FLAGS.smi_window_lengths  
    paramset3 = FLAGS.seq_window_lengths 
    lamda_set = FLAGS.lamda
    batchsz = FLAGS.batch_size  # 256

    logging("---Parameter Search-----", FLAGS)

    w = len(val_sets)
    h = len(paramset1) * len(paramset2) * len(paramset3)*len(lamda_set)

    all_predictions = [[0 for x in range(w)] for y in range(h)]
    all_losses = [[0 for x in range(w)] for y in range(h)]


    for foldind in range(len(val_sets)):
        valinds = val_sets[foldind]
        labeledinds = labeled_sets[foldind]

        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]

        train_dataset = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)

        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]

        test_dataset = prepare_interaction_pairs(XD, XT, Y, terows, tecols)

        pointer = 0

        train_loader = DataLoader(dataset=train_dataset, batch_size=batchsz, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batchsz)
        for param1value in paramset1:  # hidden neurons
            for param2value in paramset2:  # learning rate
                for param3value in paramset3:
                    for lamda in lamda_set:
                        model = net(FLAGS, param1value, param2value, param3value).cuda()
                        model.apply(weights_init)
                        rperf_list=[]
                        for epochind in range(FLAGS.num_epoch):
                            model = train(train_loader, model, FLAGS, param1value, param2value, param3value, lamda)
                            rperf, loss, rm2, auc = test(model,test_loader, FLAGS, param1value, param2value, param3value,lamda)
                            rperf_list.append(rperf)
                            ##Set the conditions for early stopping
                            if (epochind+1)%5==0:
                                print('val: epoch:{},p1:{},p2:{},p3:{},loss:{:.5f},rperf:{:.5f}, rm2:{:.5f}'.format(epochind,param1value, param2value,
                                                               param3value,loss, rperf, rm2))

                                if rperf >= max(rperf_list):
                                    torch.save(model, 'checkpoint.pth')
                                if rperf < max(rperf_list) - 0.1:
                                    print('The program is stopped early for better performance.')
                                    break
                        logging("P1 = %d,  P2 = %d, P3 = %d, Fold = %d, CI-i = %f, MSE = %f, rm2 = %f" % (
                            param1value, param2value, param3value, foldind, rperf, loss, rm2), FLAGS)

                        all_predictions[pointer][foldind] = rperf  # TODO FOR EACH VAL SET allpredictions[pointer][foldind]
                        all_losses[pointer][foldind] = loss

                    pointer += 1
    
    bestperf = -float('Inf')
    bestpointer = None

    best_param_list = []
    ##Take average according to folds, then chooose best params
    pointer = 0
    for param1value in paramset1:
        for param2value in paramset2:
            for param3value in paramset3:
                for lamda in lamda_set:
                    avgperf = 0.
                    for foldind in range(len(val_sets)):
                        foldperf = all_predictions[pointer][foldind]
                        avgperf += foldperf
                    avgperf /= len(val_sets)
        
                    if avgperf > bestperf:
                        bestperf = avgperf
                        bestpointer = pointer
                        best_param_list = [param1value, param2value, param3value,lamda,]

                    pointer += 1

    return bestpointer, best_param_list, bestperf, all_predictions, all_losses


def general_nfold_cv_test(XD, XT, Y, label_row_inds, label_col_inds, prfmeasure, FLAGS, labeled_sets, val_sets, get_rm2,
                          best_param_list,i):
    param1value = best_param_list[0]
    param2value = best_param_list[1] 
    param3value = best_param_list[2]  
    lamda = best_param_list[3]
    batchsz = FLAGS.batch_size  # 256

    logging("---Parameter Search-----", FLAGS)

    w = len(val_sets)

    all_predictions = [0 for x in range(w)] 
    all_losses = [0 for x in range(w)] 
    all_auc = [0 for x in range(w)] 
    all_aupr = [0 for x in range(w)] 
    all_preaffinities=[]
    all_affinities=[]
    for foldind in range(len(val_sets)):
        valinds = val_sets[foldind]
        labeledinds = labeled_sets[foldind]

        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]

        train_dataset = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)

        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]

        test_dataset = prepare_interaction_pairs(XD, XT, Y, terows, tecols)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batchsz, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batchsz)
        model = net(FLAGS, param1value, param2value, param3value).cuda()
        model.apply(weights_init)
        rperf_list = []
        for epochind in range(FLAGS.num_epoch):
            model = train(train_loader, model, FLAGS, param1value, param2value, param3value, lamda)
            if (epochind + 1) % 2 == 0:
                rperf, loss, rm2, auc = test(model, test_loader, FLAGS, param1value, param2value, param3value, lamda)
                rperf_list.append(rperf)
                print(
                    'test: epoch:{},p1:{},p2:{},p3:{},loss:{:.5f},rperf:{:.5f}, rm2:{:.5f}'.format(epochind, param1value,
                                                                                                  param2value,
                                                                                                  param3value, loss,
                                                                                                  rperf, rm2))

                if rperf >= max(rperf_list):
                    torch.save(model,'checkpoint.pth')
                if rperf < max(rperf_list)-0.1:
                    break
        loss_func = nn.MSELoss()
        affinities = []
        pre_affinities = []
        model=torch.load('checkpoint.pth')
        model.eval()
        for drug_SMILES, target_protein, affinity in test_loader:
            pre_affinity, _, _, _, _, _, _, _, _ = model(drug_SMILES, target_protein, FLAGS, param1value,
                                                                   param2value, param3value)
            pre_affinities += pre_affinity.cpu().detach().numpy().tolist()
            affinities += affinity.cpu().detach().numpy().tolist()

        pre_affinities = np.array(pre_affinities)
        affinities = np.array(affinities)
        if 'davis' in FLAGS.dataset_path:
            pre_label = pre_affinities
            label = np.int32(affinities>7.0)
            auc = roc_auc_score(label, pre_label)
            aupr = get_aupr(label, pre_label)
        if 'kiba' in FLAGS.dataset_path:
            pre_label = pre_affinities
            label = np.int32(affinities >12.1)
            auc = roc_auc_score(label, pre_label)
            aupr = get_aupr(label, pre_label)
        rperf = prfmeasure(affinities, pre_affinities)
        rm2 = get_rm2(affinities, pre_affinities)
        loss = loss_func(torch.Tensor(pre_affinities), torch.Tensor(affinities))
        print('best: p1:{},p2:{},p3:{},loss:{:.5f},rperf:{:.5f}, rm2:{:.5f}'.format(param1value, param2value,
                                                                                              param3value, loss, rperf,
                                                                                              rm2))

        logging("best: P1 = %d,  P2 = %d, P3 = %d, Fold = %d, CI-i = %f, MSE = %f, auc = %f, aupr = %f" % (
                    param1value, param2value, param3value, foldind, rperf, loss, auc, aupr), FLAGS)

        all_predictions[foldind] = rperf  # TODO FOR EACH VAL SET allpredictions[pointer][foldind]
        all_losses[foldind] = loss
        all_auc[foldind] = auc
        all_aupr[foldind] = aupr
        all_affinities.append(affinities)
        all_preaffinities.append(pre_affinities)
    # save affinities and preaffinites for further analysis 
    np.savetxt("./result/iter"+str(i)+"affinities.txt", np.array(all_affinities))
    np.savetxt("./result/iter"+str(i)+"preaffinities.txt", np.array(all_preaffinities))
    

    best_param_list = [param1value, param2value, param3value,lamda]
    best_perf = np.mean(all_predictions)

    return best_param_list, best_perf, all_predictions, all_losses, all_auc, all_aupr


# def plotLoss(history1, history2, history3, history4, batchind, epochind, param3ind, foldind, FLAGS):
#     figname = "b" + str(batchind) + "_e" + str(epochind) + "_" + str(param3ind) + "_" + str(foldind) + "_" + str(
#         time.time())
#     plt.figure()
#     plt.ylim(0,2)
#     plt.plot(range(FLAGS.num_epoch), history1,color='blue',label='train_loss')
#     plt.plot(range(FLAGS.num_epoch), history2,color='green',label='val_loss')
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     # plt.legend(['trainloss', 'valloss', 'cindex', 'valcindex'], loc='upper left')
#     plt.legend()
#     plt.savefig("./figures/" + figname + ".png", dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
#                 papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
#     plt.close()

#     ## PLOT CINDEX
#     plt.figure()
#     plt.title('model concordance index')
#     plt.ylabel('cindex')
#     plt.xlabel('epoch')
#     plt.plot(range(FLAGS.num_epoch), history3,color='blue',label='train_cindex')
#     plt.plot(range(FLAGS.num_epoch), history4,color='green',label='val_cindex')

#     plt.legend()
#     plt.savefig("./figures/" + figname + "_acc.png", dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
#                 papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
#     plt.close()

def prepare_interaction_pairs(XD, XT, Y, rows, cols):
    dataset = [[]]
    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        dataset[pair_ind].append(np.array(drug, dtype=np.float32))
        target = XT[cols[pair_ind]]
        dataset[pair_ind].append(np.array(target, dtype=np.float32))
        dataset[pair_ind].append(np.array(Y[rows[pair_ind], cols[pair_ind]], dtype=np.float32))
        if pair_ind < len(rows) - 1:
            dataset.append([])
    return dataset


def experiment(FLAGS, foldcount=6):  # 5-fold cross validation + test

    # Input
    # XD: [drugs, features] sized array (features may also be similarities with other drugs
    # XT: [targets, features] sized array (features may also be similarities with other targets
    # Y: interaction values, can be real values or binary (+1, -1), insert value float("nan") for unknown entries
    # perfmeasure: function that takes as input a list of correct and predicted outputs, and returns performance
    # higher values should be better, so if using error measures use instead e.g. the inverse -error(Y, P)
    # foldcount: number of cross-validation folds for settings 1-3, setting 4 always runs 3x3 cross-validation

    dataset = DataSet(fpath=FLAGS.dataset_path,  ### BUNU ARGS DA GUNCELLE
                      setting_no=FLAGS.problem_type,  ##BUNU ARGS A EKLE
                      seqlen=FLAGS.max_seq_len,
                      smilen=FLAGS.max_smi_len,
                      need_shuffle=False)
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size
    FLAGS.charsmiset_size = dataset.charsmiset_size

    XD, XT, Y = dataset.parse_data(FLAGS)

    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    drugcount = XD.shape[0]
    print(drugcount)
    targetcount = XT.shape[0]
    print(targetcount)

    FLAGS.drug_count = drugcount
    FLAGS.target_count = targetcount

    label_row_inds, label_col_inds = np.where(np.isnan(Y) == False)  

    if not os.path.exists(figdir):
        os.makedirs(figdir)
    perf = []
    mseloss = []
    auc = []
    aupr = []
    for i in range(1):
        random.seed(i+1000)
        if FLAGS.problem_type == 1:
            nfolds = get_random_folds(len(label_row_inds),foldcount)
        if FLAGS.problem_type == 2:
            nfolds = get_drugwise_folds(label_row_inds, label_col_inds, drugcount, foldcount)
        if FLAGS.problem_type == 3:
            nfolds = get_targetwise_folds(label_row_inds, label_col_inds, targetcount, foldcount)
        avgperf, avgloss, teststd, lossstd, avg_auc, auc_std, avg_aupr, aupr_std = nfold_setting_sample(XD, XT, Y, label_row_inds,label_col_inds,
                                                                            get_cindex, FLAGS, dataset, nfolds,i)
        logging("Setting " + str(FLAGS.problem_type), FLAGS)

        logging("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f, loss_std = %.5f, auc = %.5f, auc_std = %.5f, aupr =%.5f, aupr_std = %.5f" %
                    (avgperf,avgloss,teststd,lossstd, avg_auc, auc_std, avg_aupr, aupr_std), FLAGS)

        perf.append(avgperf)
        mseloss.append(avgloss)
        auc.append(avg_auc)
        aupr.append(avg_aupr)
    print(FLAGS.log_dir)

    logging(("Finally"), FLAGS)

    logging("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f, loss_std = %.5f,auc = %.5f, auc_std = %.5f, aupr =%.5f, aupr_std = %.5f" %
            (np.mean(perf), np.mean(mseloss), np.std(perf), np.std(mseloss), np.mean(auc), np.std(auc), np.mean(aupr), np.std(aupr)), FLAGS)



if __name__ == "__main__":
    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "\\"

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    logging(str(FLAGS), FLAGS)
    experiment(FLAGS)
