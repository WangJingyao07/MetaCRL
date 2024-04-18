import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np



class Learner_crl(nn.Module):
    """

    """

    def __init__(self, config, imgc, imgsz):

        super(Learner_crl, self).__init__()


        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars_crl = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn_crl = nn.ParameterList()
        
        self.fc = nn.Linear(imgc * imgsz, config['num_classes'])
        
        self.Xi = nn.Parameter(torch.Tensor(config['feature_dim'], config['num_factors']), requires_grad=True)
        nn.init.kaiming_uniform_(self.Xi, a=math.sqrt(5))
        
        self.fgr = nn.Sequential(
            nn.Linear(config['feature_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['num_factors'])
        )
    
        self.orthogonality_loss = None


        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars_crl.append(w)
                # [ch_out]
                self.vars_crl.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars_crl.append(w)
                # [ch_in, ch_out]
                self.vars_crl.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars_crl.append(w)
                # [ch_out]
                self.vars_crl.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars_crl.append(w)
                # [ch_out]
                self.vars_crl.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn_crl.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError



    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, vars=None, bn_training=True):


        if vars is None:
            vars = self.vars_crl

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)


        return x

    def disentangle_factors(self):

        self.orthogonality_loss = sum(torch.norm(self.Xi[:, i] @ self.Xi[:, j])
                                    for i in range(self.Xi.size(1) - 1)
                                    for j in range(i + 1, self.Xi.size(1)))
        
        regularization_loss = 0
        for task_i in range(self.config['N_tr']):
            avg_sample = torch.mean(self.training_samples[task_i], dim=0)
            avg_sample_rep = torch.matmul(self.Xi.T, avg_sample)
            fgr_output = self.fgr(avg_sample_rep)
            fgr_output_l1 = torch.norm(fgr_output, p=1)
            fgr_output_entropy = -torch.sum(fgr_output * torch.log(fgr_output + 1e-10))
            regularization_loss += fgr_output_l1 - fgr_output_entropy
        

        total_loss = self.config['lambda_1'] * self.orthogonality_loss + self.config['lambda_2'] * regularization_loss
        

        # self.optimizer.zero_grad()
        # total_loss.backward()
        # self.optimizer.step()
    



    # def causal_learning(self):

    #     for _ in range(self.config['outer_loop_steps']):

    #         self.meta_weights.eval() 
    #         self.causal_optimization(self.support_sets, self.query_sets)
        

    #     for _ in range(self.config['inner_loop_steps']):

    #         self.causal_optimization(self.support_sets, self.query_sets, inner_loop=True)

    #     def causal_optimization(self, support_sets, query_sets, inner_loop=False):

    #         for task_i, (support_x, support_y) in enumerate(support_sets):

    #             if inner_loop:
    #                 loss = self.inner_loop_loss(support_x, support_y)
    #             else:
    #                 loss = self.causal_loss(support_x, support_y)
                
    #             loss.backward()
    #             if not inner_loop:
    #                 self.Xi_optimizer.step()  
    #                 self.fgr_optimizer.step()  
    #             else:
    #                 self.meta_optimizer.step()  


    #         for task_i, (query_x, query_y) in enumerate(query_sets):
    #             loss = self.outer_loop_loss(query_x, query_y)
    #             loss.backward()
    #             if inner_loop:
    #                 self.meta_optimizer.step()  

        
    #         if not inner_loop:
    #             self.Xi_optimizer.zero_grad()
    #             self.fgr_optimizer.zero_grad()
    #         else:
    #             self.meta_optimizer.zero_grad()

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):

        return self.vars