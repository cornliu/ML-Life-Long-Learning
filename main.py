
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import torchvision
from torchvision import datasets, transforms

import numpy as np
import os
import random
from copy import deepcopy
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):

  def __init__(self):
    super(Model, self).__init__()
    self.fc1 = nn.Linear(3*32*32, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 256)
    self.fc4 = nn.Linear(256, 128)
    self.fc5 = nn.Linear(128, 128)
    self.fc6 = nn.Linear(128, 10)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = x.view(-1, 3*32*32)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    x = self.relu(x)
    x = self.fc4(x)
    x = self.relu(x)
    x = self.fc5(x)
    x = self.relu(x)
    x = self.fc6(x)
    return x



class EWC(object):
  """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
  """
  def __init__(self, model: nn.Module, dataloaders: list, device):
    
    self.model = model
    self.dataloaders = dataloaders
    self.device = device
    
    self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} #抓出模型的所有參數
    self._means = {} # 初始化 平均參數
    self._precision_matrices = self._calculate_importance() # 產生 EWC 的 Fisher (F) 矩陣 
    
    for n, p in self.params.items():
      self._means[n] = p.clone().detach() # 算出每個參數的平均 （用之前任務的資料去算平均）
  
  def _calculate_importance(self):
    precision_matrices = {}
    for n, p in self.params.items(): # 初始化 Fisher (F) 的矩陣（都補零）
      precision_matrices[n] = p.clone().detach().fill_(0)

    self.model.eval()
    dataloader_num=len(self.dataloaders)
    number_data = sum([len(loader) for loader in self.dataloaders])
    for dataloader in self.dataloaders:
      for data in dataloader:
        self.model.zero_grad()
        input = data[0].to(self.device)
        output = self.model(input).view(1, -1)
        label = output.max(1)[1].view(-1)
        
        ############################################################################
        #####                      產生 EWC 的 Fisher(F) 矩陣                    #####
        ############################################################################    
        loss = F.nll_loss(F.log_softmax(output, dim=1), label)             
        loss.backward()                                                    
                                                                           
        for n, p in self.model.named_parameters():                        
            precision_matrices[n].data += p.grad.data ** 2 / number_data   
                                                                   
    precision_matrices = {n: p for n, p in precision_matrices.items()}
    return precision_matrices

  def penalty(self, model: nn.Module):
    loss = 0
    for n, p in model.named_parameters():
      _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
      loss += _loss.sum()
    return loss


class MAS(object):
    """
    @article{aljundi2017memory,
      title={Memory Aware Synapses: Learning what (not) to forget},
      author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
      booktitle={ECCV},
      year={2018},
      url={https://eccv2018.org/openaccess/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf}
    }
    """
    def __init__(self, model: nn.Module, dataloaders: list, device):
        self.model = model 
        self.dataloaders = dataloaders
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} #抓出模型的所有參數
        self._means = {} # 初始化 平均參數
        self.device = device
        self._precision_matrices = self.calculate_importance() # 產生 MAS 的 Omega(Ω) 矩陣
    
        for n, p in self.params.items():
            self._means[n] = p.clone().detach()
    
    def calculate_importance(self):
        print('Computing MAS')

        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0) # 初始化 Omega(Ω) 矩陣（都補零）

        self.model.eval()
        dataloader_num = len(self.dataloaders)
        num_data = sum([len(loader) for loader in self.dataloaders])
        for dataloader in self.dataloaders:
            for data in dataloader:
                self.model.zero_grad()
                output = self.model(data[0].to(self.device))

                #######################################################################################
                #####  產生 MAS 的 Omega(Ω) 矩陣 ( 對 output 向量 算他的 l2 norm 的平方) 再取 gradient  #####
                #######################################################################################
                output.pow_(2)                                                   
                loss = torch.sum(output,dim=1)                                   
                loss = loss.mean()                                               
                loss.backward()                                                  
                                          
                for n, p in self.model.named_parameters():                      
                    precision_matrices[n].data += p.grad.abs() / num_data ## difference with EWC      

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec
class SCP(object):
    """
    OPEN REVIEW VERSION:
    https://openreview.net/forum?id=BJge3TNKwH
    """
    def __init__(self, model: nn.Module, dataloaders: list, L: int, device):
        self.model = model 
        self.dataloaders = dataloaders
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self.L= L
        self.device = device
        self._precision_matrices = self.calculate_importance()
    
        for n, p in self.params.items():
            self._means[n] = p.clone().detach()
    
    def calculate_importance(self):
        print('Computing SCP')

        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0)

        self.model.eval()
        dataloader_num = len(self.dataloaders)
        num_data = sum([len(loader) for loader in self.dataloaders])
        for dataloader in self.dataloaders:
            for data in dataloader:
                self.model.zero_grad()
                output = self.model(data[0].to(self.device))

                ####################################################################################
                #####                            TODO 區塊 （ PART 2 ）                           #####
                ####################################################################################
                ##### 產生 SCP 的 Gamma(Γ) 矩陣（ 如同 MAS 的 Omega(Ω) 矩陣, EWC 的 Fisher(F) 矩陣 ）#####
                ####################################################################################
                #####        1.對所有資料的 Output vector 取 平均 得到 平均 vector φ(:,θ_A* )       #####
                ####################################################################################
                output_mean = torch.mean(output, dim = 0)
                ####################################################################################
                #####   2. 隨機 從 單位球殼 取樣 L 個 vector ξ #（ Hint: sample_spherical() ）      #####
                ####################################################################################
                sim = sample_spherical(self.L, 10).transpose()
                sim = torch.from_numpy(sim).float().to(self.device)
                ####################################################################################
                #####   3.    每一個 vector ξ 和 vector φ( :,θ_A* )內積得到 scalar ρ               #####
                #####           對 scalar ρ 取 backward ， 每個參數得到各自的 gradient ∇ρ           #####
                #####       每個參數的 gradient ∇ρ 取平方 取 L 平均 得到 各個參數的 Γ scalar          #####  
                #####              所有參數的  Γ scalar 組合而成其實就是 Γ 矩陣                      #####
                ####(hint: 記得 每次 backward 之後 要 zero_grad 去 清 gradient, 不然 gradient會累加 )######   
                ####################################################################################
                for i in range(self.L):
                    qq = torch.dot(sim[i], output_mean)
                    qq.backward(retain_graph = True)
                    for i, j in self.model.named_parameters():
                        precision_matrices[i].data += j.grad.data**2/(self.L)
                    self.model.zero_grad()
                ####################################################################################      
                #####                            TODO 區塊 （ PART 2 ）                          #####
                ####################################################################################

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


class Convert2RGB(object):
  
  def __init__(self, num_channel):
    self.num_channel = num_channel

  def __call__(self, img):                                                                                                                                                                                                                              
    # If the channel of img is not equal to desired size,
    # then expand the channel of img to desired size.
    img_channel = img.size()[0]
    img = torch.cat([img] * (self.num_channel - img_channel + 1), 0)
    return img


class Pad(object):

  def __init__(self, size, fill=0, padding_mode='constant'):
    self.size = size
    self.fill = fill
    self.padding_mode = padding_mode
    
  def __call__(self, img):
    # If the H and W of img is not equal to desired size,
    # then pad the channel of img to desired size.
    img_size = img.size()[1]
    assert ((self.size - img_size) % 2 == 0)
    padding = (self.size - img_size) // 2
    padding = (padding, padding, padding, padding)
    return F.pad(img, padding, self.padding_mode, self.fill)

def get_transform():
  transform = transforms.Compose([transforms.ToTensor(),
                                  Pad(32),
                                  Convert2RGB(3),
                                  transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
  return transform

class Data():

  def __init__(self, path):

    transform = get_transform()

    self.MNIST_dataset = datasets.MNIST(root = os.path.join(path, "MNIST"),
                                        transform=transform,
                                        train = True,
                                        download = True)

    self.SVHN_dataset = datasets.SVHN(root = os.path.join(path, "SVHN"),
                                      transform=transform,
                                      split='train',
                                      download = True)

    self.USPS_dataset = datasets.USPS(root = os.path.join(path, "USPS"),
                                            transform=transform,
                                            train = True,
                                            download = True)
    
  def get_datasets(self):
      a = [(self.SVHN_dataset, "SVHN"),(self.MNIST_dataset, "MNIST"),(self.USPS_dataset, "USPS")]
      return a
class Dataloader():

  def __init__(self, dataset, batch_size, split_ratio=0.1):
    self.dataset = dataset[0]
    self.name = dataset[1]
    train_sampler, val_sampler = self.split_dataset(split_ratio)

    self.train_dataset_size = len(train_sampler)
    self.val_dataset_size = len(val_sampler)

    self.train_loader = data.DataLoader(self.dataset, batch_size = batch_size, sampler=train_sampler)
    self.val_loader = data.DataLoader(self.dataset, batch_size = batch_size, sampler=val_sampler)
    self.train_iter = self.infinite_iter()

  def split_dataset(self, split_ratio):
    data_size = len(self.dataset)
    split = int(data_size * split_ratio)
    indices = list(range(data_size))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = sampler.SubsetRandomSampler(train_idx)
    val_sampler = sampler.SubsetRandomSampler(valid_idx)
    return train_sampler, val_sampler
    
  def infinite_iter(self):
    it = iter(self.train_loader)
    while True:
      try:
        ret = next(it)
        yield ret
      except StopIteration:
        it = iter(self.train_loader)
def save_model(model, optimizer, store_model_path):
  # save model and optimizer
  torch.save(model.state_dict(), f'{store_model_path}.ckpt')
  torch.save(optimizer.state_dict(), f'{store_model_path}.opt')
  return

def load_model(model, optimizer, load_model_path):
  # load model and optimizer
  print(f'Load model from {load_model_path}')
  model.load_state_dict(torch.load(f'{load_model_path}.ckpt'))
  optimizer.load_state_dict(torch.load(f'{load_model_path}.opt'))
  return model, optimizer
def build_model(data_path, batch_size, learning_rate): 
  # create model
  model = Model().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  data = Data(data_path)
  datasets = data.get_datasets()
  tasks = []
  for dataset in datasets:
    tasks.append(Dataloader(dataset, batch_size))

  return model, optimizer, tasks
def normal_train(model, optimizer, task, total_epochs, summary_epochs):
  model.train()
  model.zero_grad()
  ceriation = nn.CrossEntropyLoss()
  losses = []
  loss = 0.0
  for epoch in range(summary_epochs):
    imgs, labels = next(task.train_iter)
    imgs, labels = imgs.to(device), labels.to(device)
    outputs = model(imgs)
    ce_loss = ceriation(outputs, labels)
    
    optimizer.zero_grad()
    ce_loss.backward()
    optimizer.step()

    loss += ce_loss.item()
    if (epoch + 1) % 50 == 0:
      loss = loss / 50
      print ("\r", "train task {} [{}] loss: {:.3f}      ".format(task.name, (total_epochs + epoch + 1), loss), end=" ")
      losses.append(loss)
      loss = 0.0
    
  return model, optimizer, losses
def ewc_train(model, optimizer, task, total_epochs, summary_epochs, ewc, lambda_ewc):
  model.train()
  model.zero_grad()
  ceriation = nn.CrossEntropyLoss()
  losses = []
  loss = 0.0
  for epoch in range(summary_epochs):
    imgs, labels = next(task.train_iter)
    imgs, labels = imgs.to(device), labels.to(device)
    outputs = model(imgs)
    ce_loss = ceriation(outputs, labels)
    total_loss = ce_loss
    ewc_loss = ewc.penalty(model)
    total_loss += lambda_ewc * ewc_loss 
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    loss += total_loss.item()
    if (epoch + 1) % 50 == 0:
      loss = loss / 50
      print ("\r", "train task {} [{}] loss: {:.3f}      ".format(task.name, (total_epochs + epoch + 1), loss), end=" ")
      losses.append(loss)
      loss = 0.0
    
  return model, optimizer, losses


def mas_train(model, optimizer, task, total_epochs, summary_epochs, mas_tasks, lambda_mas,alpha=0.8):
  model.train()
  model.zero_grad()
  ceriation = nn.CrossEntropyLoss()
  losses = []
  loss = 0.0
  for epoch in range(summary_epochs):
    imgs, labels = next(task.train_iter)
    imgs, labels = imgs.to(device), labels.to(device)
    outputs = model(imgs)
    ce_loss = ceriation(outputs, labels)
    total_loss = ce_loss
    mas_tasks.reverse()
    if len(mas_tasks) > 1:
        preprevious = 1 - alpha
        scalars = [alpha,preprevious]
        for mas,scalar in zip(mas_tasks[:2],scalars):
            mas_loss = mas.penalty(model)
            total_loss += lambda_mas * mas_loss * scalar
    elif len(mas_tasks) == 1:
        mas_loss = mas_tasks[0].penalty(model)
        total_loss += lambda_mas * mas_loss
    else:
        pass
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    loss += total_loss.item()
    if (epoch + 1) % 50 == 0:
      loss = loss / 50
      print ("\r", "train task {} [{}] loss: {:.3f}      ".format(task.name, (total_epochs + epoch + 1), loss), end=" ")
      losses.append(loss)
      loss = 0.0
    
  return model, optimizer, losses

def scp_train(model, optimizer, task, total_epochs, summary_epochs, scp_tasks, lambda_scp,alpha=0.65):

  model.train()
  model.zero_grad()
  ceriation = nn.CrossEntropyLoss()
  losses = []
  loss = 0.0
  for epoch in range(summary_epochs):
    imgs, labels = next(task.train_iter)
    imgs, labels = imgs.to(device), labels.to(device)
    outputs = model(imgs)
    ce_loss = ceriation(outputs, labels)
    total_loss = ce_loss
    scp_tasks.reverse()
    if len(scp_tasks) > 1:
        preprevious = 1 - alpha
        scalars = [alpha,preprevious]
        for scp,scalar in zip(scp_tasks[:2],scalars):
            scp_loss = scp.penalty(model)
            total_loss += lambda_scp * scp_loss * scalar
    elif len(scp_tasks) == 1:
        scp_loss = scp_tasks[0].penalty(model)
        total_loss += lambda_scp * scp_loss
    else:
        pass
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    loss += total_loss.item()
    if (epoch + 1) % 50 == 0:
      loss = loss / 50
      print ("\r", "train task {} [{}] loss: {:.3f}      ".format(task.name, (total_epochs + epoch + 1), loss), end=" ")
      losses.append(loss)
      loss = 0.0
  ###############################
  #####  TODO 區塊 （PART 2） #####
  ###############################
  ##  參考 MAS. EWC train 的寫法 ##                 
  ###############################
  #####  TODO 區塊 （PART 2） #####
  ###############################
  return model, optimizer, losses

def val(model, task):
  model.eval()
  correct_cnt = 0
  for imgs, labels in task.val_loader:
    imgs, labels = imgs.to(device), labels.to(device)
    outputs = model(imgs)
    _, pred_label = torch.max(outputs.data, 1)

    correct_cnt += (pred_label == labels.data).sum().item()
    
  return correct_cnt / task.val_dataset_size

def train_process(model, optimizer, tasks, config):
  task_loss, acc = {}, {}
  for task_id, task in enumerate(tasks):
    print ('\n')
    total_epochs = 0
    task_loss[task.name] = []
    acc[task.name] = []
    if config.mode == 'basic' or task_id == 0:
      while (total_epochs < config.num_epochs):
        model, optimizer, losses = normal_train(model, optimizer, task, total_epochs, config.summary_epochs)
        task_loss[task.name] +=  losses

        for subtask in range(task_id + 1):
          acc[tasks[subtask].name].append(val(model, tasks[subtask]))

        total_epochs += config.summary_epochs
        if total_epochs % config.store_epochs == 0 or total_epochs >= config.num_epochs:
          save_model(model, optimizer, config.store_model_path)
    
    if config.mode == 'ewc' and task_id > 0:
      old_dataloaders = []
      for old_task in range(task_id): 
        old_dataloaders += [tasks[old_task].val_loader]
      ewc = EWC(model, old_dataloaders, device)
      while (total_epochs < config.num_epochs):
        model, optimizer, losses = ewc_train(model, optimizer, task, total_epochs, config.summary_epochs, ewc, config.lifelong_coeff)
        task_loss[task.name] +=  losses

        for subtask in range(task_id + 1):
          acc[tasks[subtask].name].append(val(model, tasks[subtask]))

        total_epochs += config.summary_epochs
        if total_epochs % config.store_epochs == 0 or total_epochs >= config.num_epochs:
          save_model(model, optimizer, config.store_model_path)

    if config.mode == 'mas' and task_id > 0:
      old_dataloaders = []
      mas_tasks = []
      for old_task in range(task_id): 
        old_dataloaders += [tasks[old_task].val_loader]
        mas = MAS(model, old_dataloaders, device) #要改 
        mas_tasks += [mas]
      while (total_epochs < config.num_epochs):
        model, optimizer, losses = mas_train(model, optimizer, task, total_epochs, config.summary_epochs, mas_tasks, config.lifelong_coeff)
        task_loss[task.name] +=  losses

        for subtask in range(task_id + 1):
          acc[tasks[subtask].name].append(val(model, tasks[subtask]))

        total_epochs += config.summary_epochs
        if total_epochs % config.store_epochs == 0 or total_epochs >= config.num_epochs:
          save_model(model, optimizer, config.store_model_path)

    if config.mode == 'scp' and task_id > 0:         ################PART 2###############
      old_dataloaders = []
      scp_tasks = []
      for old_task in range(task_id): 
        old_dataloaders += [tasks[old_task].val_loader]
        scp = SCP(model, old_dataloaders, 100,device) #要改 
        scp_tasks += [scp]
      while (total_epochs < config.num_epochs):
        model, optimizer, losses = scp_train(model, optimizer, task, total_epochs, config.summary_epochs, scp_tasks, config.lifelong_coeff)
        task_loss[task.name] +=  losses

        for subtask in range(task_id + 1):
          acc[tasks[subtask].name].append(val(model, tasks[subtask]))

        total_epochs += config.summary_epochs
        if total_epochs % config.store_epochs == 0 or total_epochs >= config.num_epochs:
          save_model(model, optimizer, config.store_model_path)
      ########################################
      ##       TODO 區塊 （ PART 2 ）         ##
      ########################################
      ##    PART 2  implementation 的部份    ##
      ##   你也可以寫別的 regularization 方法  ##
      ##    助教這裡有提供的是  scp    的 作法   ##
      ##     Slicer Cramer Preservation     ##
      ########################################
      ########################################
      ##       TODO 區塊 （ PART 2 ）         ##
      ########################################
  return task_loss, acc

class configurations(object):
  def __init__(self):
    self.batch_size = 256
    self.num_epochs = 10000
    self.store_epochs = 250
    self.summary_epochs = 250
    self.learning_rate = 0.0005
    self.load_model = False
    self.store_model_path = "./model"
    self.load_model_path = "./model"
    self.data_path = "./data"
    self.mode = None
    self.lifelong_coeff = 0.5

###### 你也可以自己設定參數   ########
###### 但上面的參數 是這次作業的預設直 #########

"""
the order is svhn -> mnist -> usps
===============================================

"""
# import tqdm

if __name__ == '__main__':
    mode_list = ['mas','ewc','basic','scp']
    # mode_list = ['scp']

    ## hint: 謹慎的去選擇 lambda 超參數 / ewc: 80~400, mas: 0.1 - 10
    ############################################################################
    #####                           TODO 區塊 （ PART 1 ）                   #####
    ############################################################################ 
    coeff_list = [1, 400 ,0 , 15]  ## 你需要在這 微調 lambda 參數, mas, ewc, baseline=0##  
    ############################################################################
    #####                           TODO 區塊 （ PART 1 ）                   #####
    ############################################################################
    
    config = configurations()
    count = 0
    for mode in mode_list:
        config.mode = mode
        config.lifelong_coeff = coeff_list[count]
        print("{} training".format(config.mode))    
        model, optimizer, tasks = build_model(config.load_model_path, config.batch_size, config.learning_rate)
        print ("Finish build model")
        if config.load_model:
            model, optimizer = load_model(model, optimizer, config.load_model_path)
        task_loss, acc = train_process(model, optimizer, tasks, config)
        with open(f'./{config.mode}_acc.txt', 'w') as f:
            json.dump(acc, f)
        count += 1

# %matplotlib inline
import matplotlib.pyplot as plt



def plot_result(mode_list, task1, task2, task3):
  
    #draw the lines
    count = 0
    for reg_name in mode_list:
        label = reg_name
        with open(f'./{reg_name}_acc.txt', 'r') as f:
            acc = json.load(f)
        if count == 0: 
            color= 'red'
        elif count  == 1:
            color= 'blue'
        elif count == 2:
            color = 'yellow'
        else:
            color = 'purple'
        ax1=plt.subplot(3, 1, 1)
        plt.plot(range(len(acc[task1])),acc[task1],color,label=label)
        ax1.set_ylabel(task1)
        ax2=plt.subplot(3, 1, 2,sharex=ax1,sharey=ax1)
        plt.plot(range(len(acc[task3]),len(acc[task1])),acc[task2],color,label=label)
        ax2.set_ylabel(task2)
        ax3=plt.subplot(3, 1, 3,sharex=ax1,sharey=ax1)
        ax3.set_ylabel(task3)
        plt.plot(range(len(acc[task2]),len(acc[task1])),acc[task3],color,label=label)
        count += 1
    plt.ylim((0.2,1.02))
    plt.legend()
    plt.show()
    return 

mode_list = ['ewc','mas','basic', 'scp']
plot_result(mode_list,'SVHN','MNIST','USPS')