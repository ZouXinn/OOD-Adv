from .universalattack import UniversalAttack, UAP, AverageMeter
import torch
import torch.nn as nn
import time

class SimpleUAP(UniversalAttack):
    r'''
    the SimpleUAP method in paper "Towards Data-free Universal Adversarial Perturbations With Artifical Jigsaw Images"
    [https://papersnlp.github.io/rubustml/99_CameraReady_ICLR_2021_Workshop_Towards%20a%20Data-free%20Universal%20Adversarial%20Attack_v2.pdf]
    '''
    def __init__(self, model, eps=8/255, lr=0.01, n_iters=3000, delta=0.2, print_freq=100):
        super().__init__("SimpleUAP", model)
        self.eps = eps
        self.lr = lr
        self.n_iters = n_iters
        self.delta = delta
        self.print_freq = print_freq
        

    def train(self, uap: UAP, dataloader):
        optimizer = torch.optim.Adam(uap.parameters(), lr=self.lr)
        ori_training = self.model.training
        self.model.eval()
        cosine = nn.CosineSimilarity()
        
        batch_time = AverageMeter()
        losses = AverageMeter()
        
        data_iterator = iter(dataloader)
        iteration = 0
        end = time.time()
        while iteration < self.n_iters:
            try:
                x, y = next(data_iterator)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                data_iterator = iter(dataloader)
                x, y = next(data_iterator)
            x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            output_adv = self.model(uap(x))
            
            
            loss = cosine(output, output_adv)
            loss = loss.mean()
            
            losses.update(loss.item(), x.size(0))
            
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #### remenber to clip the UAP
            uap.uap.data = torch.clamp(uap.uap.data, -self.eps, self.eps)
            
            batch_time.update(time.time() - end)
            end = time.time()
            if iteration % self.print_freq == 0:
                print(f"iter {iteration}, batch time {batch_time.avg}, loss {losses.avg}")
            
            iteration+=1
        
        print(f"iter {iteration}, batch time {batch_time.avg}, loss {losses.avg}")
        if ori_training:
            self.model.train()
        return uap