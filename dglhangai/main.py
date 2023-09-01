import torch
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from utils import load_data, EarlyStopping
import random
import os
import shutil
from 生成标签 import biaoqian

def Combinations(L, k):
    """List all combinations: choose k elements from list L"""
    na = len(L)
    result = [] # To Place Combination result
    for i in range(na-k+1):
        if k > 1:
            newL = L[i+1:]
            Comb, _ = Combinations(newL, k - 1)
            for item in Comb:
                item.insert(0, L[i])
                result.append(item)
        else:
            result.append([L[i]])
    return result, len(result)

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1

def evaluate1(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1

def softmax(X):
    X_exp = torch.exp(X)#对元素进行指数计算
    partition = X_exp.sum(1, keepdim=True)#对每一行进行求和
    return X_exp / partition

def predict(model, g, features, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    pp = softmax(logits[mask])
    _, indices = torch.max(logits[mask], dim=1)
    prediction = indices.long().cpu().numpy()

    return pp

def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.

    wai = 0
    for mmh in range(1):
        # if wai==0:
        #     biaoqian()
        path_0 = r'G:\图神经网络编程\有向异构\dglhan改\结果\save' + str(wai)
        if os.path.exists(path_0):
            shutil.rmtree(path_0)
        os.makedirs(path_0)

        g, features, labels, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask = load_data(args['dataset'])

        num_classes = 2

        if hasattr(torch, 'BoolTensor'):
            train_mask = train_mask.bool()
            val_mask = val_mask.bool()
            test_mask = test_mask.bool()

        features = features.to(args['device'])
        labels = labels.to(args['device'])
        train_mask = train_mask.to(args['device'])
        val_mask = val_mask.to(args['device'])
        test_mask = test_mask.to(args['device'])
        train_maskk = train_mask
        val_maskk = val_mask

        if args['hetero']:
            print('aaaaa')
            from model_hetero import HAN
            model = HAN(meta_paths=[['pa', 'ap'], ['pf', 'fp']],
                        in_size=features.shape[1],
                        hidden_size=args['hidden_units'],
                        out_size=num_classes,
                        num_heads=args['num_heads'],
                        dropout=args['dropout']).to(args['device'])
            g = g.to(args['device'])
        else:
            print('bbbbb')
            from model import HAN
            model = HAN(num_meta_paths=len(g),
                        in_size=features.shape[1],
                        hidden_size=args['hidden_units'],
                        out_size=num_classes,
                        num_heads=args['num_heads'],
                        dropout=args['dropout']).to(args['device'])
            g = [graph.to(args['device']) for graph in g]

        stopper = EarlyStopping(patience=args['patience'])
        loss_fcn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                     weight_decay=args['weight_decay'])
        optimizer2 = torch.optim.Adam(model.parameters(), lr=args['lr2'],
                                     weight_decay=args['weight_decay2'])
        #——————元模块——————
        step = 10  # 根据想要生成的元图个数来定

        train_shot_0 = 5
        train_shot_1 = 20

        select_class = [0, 1]
        class1_idx = []
        class2_idx = []
        train_idx_i = train_idx

        y = labels.tolist()
        r0 = []
        r1 = []
        r2 = []
        for uuu in y:
            if int(uuu)==0:
                r0.append(1)
                r1.append(0)
                r2.append(0)
            elif int(uuu)==1:
                r1.append(1)
                r0.append(0)
                r2.append(0)
            elif int(uuu)==2:
                r2.append(1)
                r0.append(0)
                r1.append(0)

        p = pd.DataFrame()
        p[0] = r0
        p[1] = r1
        p[2] = r2
        truelabels = np.array(p)
        #truelabels = np.array(truelabelss)

        n1_ = Combinations(select_class, 2)
        ee = 0
        labels_local = pd.DataFrame()
        labels_local[0] = list(0 for r in range(len(truelabels)))
        labels_local[1] = list(0 for r in range(len(truelabels)))
        labels_local[2] = list(0 for r in range(len(truelabels)))
        labels_local = np.array(labels_local)
        for ru in train_idx:
            ru = int(ru)
            labels_local[ru] = truelabels[ru]
        for select_class_i in n1_[0]:
            se = select_class_i[0]
            se1 = select_class_i[1]
            for k in train_idx_i:
                k = int(k)
                if (int(pd.DataFrame(labels_local)[se][k]) == 1):
                    class1_idx.append(k)
                elif (int(pd.DataFrame(labels_local)[se1][k]) == 1):
                    class2_idx.append(k)
            for m in range(step):
                # if m!=0:
                #     stopper.load_checkpoint(model)
                # else:
                #     sdf = 2
                class1_train = random.sample(class1_idx, train_shot_0)
                class2_train = random.sample(class2_idx, train_shot_1)
                class1_val = [n1 for n1 in class1_idx if n1 not in class1_train]
                class2_val = [n2 for n2 in class2_idx if n2 not in class2_train]

                train_idx2 = class1_train+class2_train
                #random.shuffle(train_idx)
                val_idx2 = class1_val+class2_val
                #random.shuffle(val_idx)

                #y = labels_local
                train_idx2 = np.array(train_idx2)
                val_idx2 = np.array(val_idx2)

                #train2 = np.array([x for x in train_idx.tolist() if x not in range(len(train_idx2))])
                #val2 = np.array([x for x in val_idx.tolist() if x not in range(len(val_idx2))])

                train_mask = pd.DataFrame()
                train_mask[0] = [0 for r in range(len(truelabels))]
                train_mask = np.array(train_mask)
                for ru in train_idx2:
                    ru = int(ru)
                    train_mask[ru] = True
                train_mask = train_mask.reshape(1, len(train_mask)).tolist()[0]
                train_mask = torch.tensor(train_mask).bool()
                train_mask = train_mask.to(args['device'])
                #train_mask = pd.DataFrame(train_mask)

                val_mask = pd.DataFrame()
                val_mask[0] = [0 for r in range(len(truelabels))]
                val_mask = np.array(val_mask)
                for ru in val_idx2:
                    ru = int(ru)
                    val_mask[ru] = int(1)
                val_mask = val_mask.reshape(1,len(val_mask)).tolist()[0]
                val_mask = torch.tensor(val_mask).bool()
                val_mask = val_mask.to(args['device'])

                for epoch in range(args['num_epochs']):
                    model.train()
                    logits = model(g, features)
                    loss = loss_fcn(logits[train_mask], labels[train_mask])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
                    val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn)
                    early_stop = stopper.step(val_loss.data.item(), val_acc, model)

                    print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
                          'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
                        epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))
                stopper.save_checkpoint(model)
                    # if early_stop:
                    #     #stopper.save_checkpoint(model)
                    #     break
        uuu = 0
        stopper.load_checkpoint(model)
        print('aaaaaaaaaaaaaaaaaaaaaaa')
        for epoch1 in range(args['num_epochs1']):
            model.train()
            logits = model(g, features)
            loss2 = loss_fcn(logits[train_maskk], labels[train_maskk])

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            train_acc, train_micro_f11, train_macro_f11 = score(logits[train_maskk], labels[train_maskk])
            val_loss1, val_acc1, val_micro_f11, val_macro_f11 = evaluate1(model, g, features, labels, val_maskk, loss_fcn)
            early_stop = stopper.step(val_loss1.data.item(), val_acc1, model)

            print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
                  'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
                epoch1 + 1, loss2.item(), train_micro_f11, train_macro_f11, val_loss1.item(), val_micro_f11, val_macro_f11))
            if float(val_loss1.item())<0.05:
                uuu = uuu+1
            if uuu>3:
                break
            # if early_stop:
            #     stopper.save_checkpoint(model)
            #     break
        stopper.save_checkpoint(model)

        stopper.load_checkpoint(model)
        pred = predict(model, g, features, test_mask)
        pred1 = pred.tolist()
        # test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, features, labels, test_mask, loss_fcn)
        # print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        #     test_loss.item(), test_micro_f1, test_macro_f1))

        # 自动把每一次结果按照概率从高到低排序，选取最后的几个作为下一次的负标签
        bap = pd.read_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\标签.xlsx', index_col=None)
        #bap = pd.DataFrame(pd.concat([ap[0], ap[1]]).unique())
        test_prel = pd.DataFrame(pred1)

        jieguo = pd.DataFrame()
        jieguo[0] = bap[0]
        jieguo[1] = test_prel[0]
        jieguo[2] = test_prel[1]
        jieguo.columns = ['1', '2', '3']
        jieguo.sort_values('2', inplace=True, ascending=False)
        jieguo.to_excel(r'G:\图神经网络编程\有向异构\dglhan改\结果\save' + str(wai) + '\\' + '预测结果'+str(wai) + '.xlsx')

        # a = pd.read_excel(r'G:\图神经网络编程\有向异构\dglhan改\结果\save' + str(wai) + '\\' + '预测结果'+str(wai) + '.xlsx', header=None)
        # b = pd.read_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\标签.xlsx', index_col=None)
        # #正标签的名字
        # bbb = b[0][b[2]==1].tolist()
        # # c = random.sample(list(a[1])[-50:],6)#规定负标签选取多少数量的
        # jieguo.sort_values('3', inplace=True, ascending=False)
        # cf = jieguo['1'].tolist()
        # c = []
        # for fo in cf:
        #     if len(c)<100:#调节负标签的数量
        #         if fo not in bbb:
        #             c.append(fo)
        #         else:
        #             continue
        #     else:
        #         break
        #
        # # 每次迭代新的负标签
        # lii = []
        # for yu, yi in enumerate(b[0].tolist()):
        #     if yi in c:
        #         lii.append(1)
        #     else:
        #         lii.append(0)
        # lii = pd.DataFrame(lii)
        #
        # t1 = b[b[2] == 1].index.tolist()
        # t2 = lii[lii[0] == 1].index.tolist()
        # ls = []#无标签的(标记为2的)
        # for yu in range(len(b)):
        #     if yu in t1:
        #         ls.append(int(0))
        #     elif yu in t2:
        #         ls.append(int(0))
        #     else:
        #         ls.append(int(1))
        #
        # d = pd.DataFrame()
        # d[0] = b[0].tolist()
        # d[1] = [j for j in range(len(b))]
        # d[2] = b[2].tolist()#正标签
        # d[3] = lii#新的负标签
        # d[4] = ls#无标签的(标记为2的)
        # d.to_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\标签.xlsx', index=None)
        wai = wai+1

if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
