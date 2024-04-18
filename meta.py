import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner.learner import Learner
from    copy import deepcopy
from    learnerCRL import Learner_crl


class Meta(nn.Module):
    def __init__(self, args, config):

        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.xi_lr = args.xi_lr
        self.gr_lr = args.gr_lr

        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num

        self.meta_update_step = args.meta_update_step
        self.meta_update_step_test = args.meta_update_step_test
        self.crl_update_step = args.crl_update_step
        self.crl_update_step_test = args.crl_update_step_test

        self.net = Learner(config, args.imgc, args.imgsz)
        self.crlnet = Learner_crl(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)


    def clip_grad_by_norm_(self, grad, max_norm):
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry):

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.meta_update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.meta_update_step + 1)]

        gen_rep = torch.matmul(self.Xi.T, x)
        
        
        avg_sample = torch.mean(x, dim=0)  
        avg_sample_rep = torch.matmul(self.Xi.T, avg_sample)  
        # fgr_output = torch.sigmoid(self.fgr(avg_sample_rep))  

        

        logits = self.fc(gen_rep)

        for i in range(task_num):

            logits = self.crlnet(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            crl_grad = torch.autograd.grad(loss, self.crlnet.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            crl_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(crl_grad, self.crlnet.parameters())))

            with torch.no_grad():

                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            with torch.no_grad():

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct


            for k in range(1, self.meta_update_step):

                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])

                grad = torch.autograd.grad(loss, fast_weights)

                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)

                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct

            loss_q = losses_q[-1] / task_num

            self.meta_optim.zero_grad()
            loss_q.backward()

            self.meta_optim.step()

            acc = np.array(corrects) / (querysz * task_num)



        return acc



    def finetunning(self, x_spt, y_spt, x_qry, y_qry):

        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.meta_update_step_test + 1)]

        net = deepcopy(self.net)

        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        with torch.no_grad():
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        with torch.no_grad():
            logits_q = net(x_qry, fast_weights, bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.meta_update_step_test):
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)

            grad = torch.autograd.grad(loss, fast_weights)

            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)

            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[k + 1] = corrects[k + 1] + correct

        acc = np.array(corrects) / querysz

        return acc


def main():
    pass


if __name__ == '__main__':
    main()
