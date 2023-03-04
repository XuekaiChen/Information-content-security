import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert.optimization import BertAdam


def train(train_iter, dev_iter, model, args, valid_num):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight': 0.0}]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.lr,
                         warmup=0.05,
                         t_total=len(train_iter) * args.epochs)

    best_acc = 0
    best_epoch = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        print('\n-----------epochs: {}-----------'.format(epoch))
        batch_step = 0
        for batch in train_iter:
            batch_step += 1
            samples, target = batch
            if samples[0].shape == torch.Size([0]):
                continue
            optimizer.zero_grad()
            logit = model(samples)

            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            if batch_step % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = corrects.item() / args.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss:{:.6f} acc:{:.4f}({}/{})'.format(
                        batch_step, loss.item(), accuracy, corrects, args.batch_size
                    )
                )

        # 一个epoch结束，evaluate
        dev_acc, dev_loss = data_eval(dev_iter, model, args, valid_num)
        if epoch > 5 and dev_loss > 0.9:
            print('\nloss does not converge. validation loss is {}, training done...'.format(dev_loss))
            return
        if dev_acc >= best_acc:
            best_acc = dev_acc
            best_epoch = epoch
            save(model, args.save_dir, 'best', epoch)
        if epoch - best_epoch >= args.early_stop:
            print('early stop by {} epochs.'.format(epoch))
            return

        model.train()


def data_eval(data_iter, model, args, eval_num):
    model.eval()
    corrects, avg_loss = 0, 0
    batch_num = 0
    logits = None
    targets = []
    for batch in data_iter:
        samples, target = batch
        with torch.no_grad():
            logit = model(samples)

        if logits is None:
            logits = logit
        else:
            logits = torch.cat([logits, logit], 0)

        targets.extend(target.tolist())

        loss = F.cross_entropy(logit, target)
        batch_num += 1
        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    avg_loss /= batch_num
    accuracy_ = corrects.item() / eval_num
    if not args.test:  # validation phase
        print('\nValidation - loss:{:.6f} acc:{:.4f}({}/{})'.format(
            avg_loss, accuracy_, corrects, eval_num))
        return accuracy_, avg_loss

    else:  # testing phase
        # mertrics
        from sklearn import metrics
        predictions = torch.max(logits, 1)[1].cpu().detach().numpy()
        labels = np.array(targets)
        accuracy = metrics.accuracy_score(labels, predictions)
        precious = metrics.precision_score(labels, predictions)
        recall = metrics.recall_score(labels, predictions)
        F1_score = metrics.f1_score(labels, predictions)
        confusion_matrix = metrics.confusion_matrix(labels, predictions)
        print('Testing - loss:{:.6f} acc:{:.4f}({}/{})'.format(avg_loss, accuracy, corrects, eval_num))
        print("confusion matrix: \n{}\n".format(confusion_matrix))
        result_file = os.path.join(args.save_dir, 'result_new.txt')
        with open(result_file, 'a', errors='ignore') as f:
            f.write('The testing accuracy: {:.4f} \n'.format(accuracy))
            f.write('The testing precious: {:.4f} \n'.format(precious))
            f.write('The testing recall: {:.4f} \n'.format(recall))
            f.write('The testing F1_score: {:.4f} \n'.format(F1_score))
            f.write('The testing confusion matrix: \n{} \n\n'.format(confusion_matrix))
        return accuracy, recall, precious, F1_score


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_epochs_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
