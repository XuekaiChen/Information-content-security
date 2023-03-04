import os
import sys
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from tensorboardX import SummaryWriter


def train(train_set, dev_set, model, args, valid_num):
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-6)
    best_acc = 0
    best_epoch = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        print('\n-----------epochs: {}-----------'.format(epoch))
        batch_step = 0
        for idx in range(0, len(train_set[0]), args.batch_size):
            batch_step += 1
            input_lists = train_set[0][idx: idx + args.batch_size]
            target_lists = train_set[2][idx: idx + args.batch_size]
            input_tensors = torch.LongTensor(input_lists).to(args.device)
            target_tensors = torch.LongTensor(target_lists).to(args.device)
            logit = model(input_tensors)
            loss = F.cross_entropy(logit, target_tensors)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_step % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target_tensors.size()).data == target_tensors.data).sum()
                accuracy = corrects.item() / args.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss:{:.6f} acc:{:.4f}({}/{})'.format(
                        batch_step, loss.item(), accuracy, corrects, args.batch_size
                    )
                )

        # 一个epoch结束，evaluate
        dev_acc, dev_loss = data_eval(dev_set, model, args, valid_num)
        if epoch > 10 and dev_loss > 0.7:
            print('\nloss does not converge. validation loss is {}, training done...'.format(dev_loss))
            return
        if dev_acc > best_acc:
            best_acc = dev_acc
            best_epoch = epoch
            save(model, args.save_dir, 'best', epoch)
        if epoch - best_epoch >= args.early_stop:
            print('early stop by {} epochs.'.format(epoch))
            return

        model.train()


def data_eval(mode_set, model, args, eval_num):
    model.eval()
    corrects, avg_loss = 0, 0
    batch_num = 0
    logits = None
    targets = []
    with torch.no_grad():
        for idx in range(0, len(mode_set[0]), args.batch_size):
            input_lists_ = mode_set[0][idx: idx + args.batch_size]
            target_lists_ = mode_set[2][idx: idx + args.batch_size]
            input_tensors_ = torch.LongTensor(input_lists_).to(args.device)
            target_tensors_ = torch.LongTensor(target_lists_).to(args.device)
            logit = model(input_tensors_)

            if logits is None:
                logits = logit
            else:
                logits = torch.cat([logits, logit], 0)
            targets.extend(target_lists_)

            loss = F.cross_entropy(logit, target_tensors_)
            batch_num += 1
            avg_loss += loss.item()
            corrects += (torch.max(logit, 1)[1].view(target_tensors_.size()).data == target_tensors_.data).sum()

    avg_loss /= batch_num
    accuracy_ = corrects.item() / eval_num
    if not args.test:  # validation phase
        print('\nValidation - loss:{:.6f} acc:{:.4f}({}/{})'.format(
            avg_loss, accuracy_, corrects, eval_num))
        return accuracy_, avg_loss

    else:  # testing phase
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
        result_file = os.path.join(args.save_dir, 'result.txt')
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

