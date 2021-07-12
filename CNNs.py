import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable  
from graphviz import Digraph
from collections import OrderedDict

class Lstm(nn.Module):
    def __init__(self, num_sizes, num_classes):
        super(Lstm,self).__init__()
        self.LSTM = nn.LSTM(num_sizes,128,batch_first=True,num_layers=3)
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,num_classes)
    def forward(self,x):
        out, (h_n, c_n) = self.LSTM(x)
        out = self.fc1(out[:,-1,:])
        out = self.fc2(out)
        return out

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    def forward(self,x):
        out=F.relu(self.conv1(x))
        out=F.max_pool2d(out,2)
        out=F.relu(self.conv2(out))
        out=F.max_pool2d(out,2)
        out=out.view(out.size(0),-1)
        out1=F.relu(self.fc1(out))
        out2=F.relu(self.fc2(out1))
        out3=self.fc3(out2)
        return out3

class AlexNet(nn.Module):
    def __init__(self,num_classes):
        super(AlexNet,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,96,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(96,256,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(256,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        self.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,num_classes),
        )
    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),256*6*6)
        x=self.classifier(x)
        return x

cfg={
    'VGG11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'VGG13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'VGG16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'VGG19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M'],
}
class VGG(nn.Module):
    def __init__(self,vgg_name):
        super(VGG,self).__init__()
        self.features=self._make_layers(cfg[vgg_name])
        self.classifier=nn.Linear(512,10)
    def forward(self,x):
        out=self.features(x)
        out=out.view(out.size(0),-1)
        out=self.classifier(out)
        return out
    def _make_layers(self,cfg):
        layers=[]
        in_channels=3
        for x in cfg:
            if x =='M':
                layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                layers+=[nn.Conv2d(in_channels,x,kernel_size=3,padding=1),nn.BatchNorm2d(x),nn.ReLU(inplace=True)]
                in_channels=x
            layers+=[nn.AvgPool2d(kernel_size=1,stride=1)]
        return nn.Sequential(*layers)
        
# 编写卷积+bn+relu模块
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channals, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channals, **kwargs)
        self.bn = nn.BatchNorm2d(out_channals)
 
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)
 
# 编写Inception模块
class Inception(nn.Module):
    def __init__(self, in_planes,
                 n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = BasicConv2d(in_planes, n1x1, kernel_size=1)
 
        # 1x1 conv -> 3x3 conv branch
        self.b2_1x1_a = BasicConv2d(in_planes, n3x3red,
                                    kernel_size=1)
        self.b2_3x3_b = BasicConv2d(n3x3red, n3x3,
                                    kernel_size=3, padding=1)
 
        # 1x1 conv -> 3x3 conv -> 3x3 conv branch
        self.b3_1x1_a = BasicConv2d(in_planes, n5x5red,
                                    kernel_size=1)
        self.b3_3x3_b = BasicConv2d(n5x5red, n5x5,
                                    kernel_size=3, padding=1)
        self.b3_3x3_c = BasicConv2d(n5x5, n5x5,
                                    kernel_size=3, padding=1)
 
        # 3x3 pool -> 1x1 conv branch
        self.b4_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.b4_1x1 = BasicConv2d(in_planes, pool_planes,
                                  kernel_size=1)
 
    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2_3x3_b(self.b2_1x1_a(x))
        y3 = self.b3_3x3_c(self.b3_3x3_b(self.b3_1x1_a(x)))
        y4 = self.b4_1x1(self.b4_pool(x))
        # y的维度为[batch_size, out_channels, C_out,L_out]
        # 合并不同卷积下的特征图
        return torch.cat([y1, y2, y3, y4], 1)
 
    
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = BasicConv2d(3, 192,
                                      kernel_size=3, padding=1)
 
        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
 
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
 
        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
 
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
 
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)
 
    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def make_dot(var, params=None):
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}
 
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
 
    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'
 
    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot
'''
class ANN_O(nn.Module):
    def __init__(self, num, num_classes):
        super(ANN_O, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6*num,500),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(500,500),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(500,500),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(500,num_classes)
        )
    def forward(self, x):
        
        x = x.view(x.size(0),-1)
        out = self.fc(x)
        
        return out
'''   
class ANN_O(nn.Module):
    def __init__(self, N, num_classes):
        super(ANN_O, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(N,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU(inplace=True),
            nn.Conv2d(16,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc3 = nn.Linear(64,num_classes)
        
    def forward(self, x):
        
        x = x.view(x.size(0), 1, 10, 6)
        out = self.features(x)
        
        out = out.view(out.size(0), -1)
        out_1 = self.fc1(out)
        out_2 = self.fc2(out_1)
        out_3 = self.fc3(out_2)
        
        return out_3

class ANN_T(nn.Module):
    def __init__(self, N, num_classes):
        super(ANN_T, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(N,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU(inplace=True),
            nn.Conv2d(16,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(96,128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc3 = nn.Linear(64,num_classes)
        
    def forward(self, x):
        
        out = self.features(x)
        
        out = out.view(out.size(0), -1)
        out_1 = self.fc1(out)
        out_2 = self.fc2(out_1)
        out_3 = self.fc3(out_2)
        
        return out_3

class ANN_BOT(nn.Module):
    def __init__(self, num_classes):
        super(ANN_BOT, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc3 = nn.Linear(256,num_classes)
        
    def forward(self, x):

        out_1 = self.fc1(x)
        out_2 = self.fc2(out_1)
        out_3 = self.fc3(out_2)
        
        return out_3

class ANN_OTF(nn.Module):
    def __init__(self, num_classes):
        super(ANN_OTF, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(96+64,256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc3 = nn.Linear(256,num_classes)
        
        self.fc11 = nn.Sequential(
            nn.Linear(64,256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc12 = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc13 = nn.Linear(256,10)
        
        self.fc21 = nn.Sequential(
            nn.Linear(96,256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc22 = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc23 = nn.Linear(256,10)
        
    def forward(self, x1, x2):
        
        n = self.fc13(self.fc12(self.fc11(x1)))
        m = self.fc23(self.fc22(self.fc21(x2)))
        N = n.argmax(dim=1) + 1
        M = m.argmax(dim=1) + 1

        x1 = torch.mul(x1,N.reshape(-1,1).float())
        x2 = torch.mul(x2,M.reshape(-1,1).float())
        
        x = torch.cat([x1, x2], 1)

        out_1 = self.fc1(x)
        out_2 = self.fc2(out_1)
        out_3 = self.fc3(out_2)
        
        return out_3

class ANN_OT(nn.Module):
    def __init__(self, N, M, num_classes):
        super(ANN_OT, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(N,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU(inplace=True),
            nn.Conv2d(16,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(M,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU(inplace=True),
            nn.Conv2d(16,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(96+64,256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc3 = nn.Linear(256,num_classes)
        
        self.fc11 = nn.Sequential(
            nn.Linear(64,256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc12 = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc13 = nn.Linear(256,100)
        
        self.fc21 = nn.Sequential(
            nn.Linear(96,256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc22 = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc23 = nn.Linear(256,100)
        
    def forward(self, x1, x2):
        x1 = x1.view(x1.size(0), 1, 10, 6)
        x1 = self.features1(x1)
        x2 = self.features2(x2)
        
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        
        n = self.fc13(self.fc12(self.fc11(x1)))
        m = self.fc23(self.fc22(self.fc21(x2)))
        N = n.argmax(dim=1) + 1
        M = m.argmax(dim=1) + 1

        X1 = torch.mul(x1,N.reshape(-1,1).float())
        X2 = torch.mul(x2,M.reshape(-1,1).float())
        
        x = torch.cat([x1, x2], 1)

        out_1 = self.fc1(x)
        out_2 = self.fc2(out_1)
        out_3 = self.fc3(out_2)
        
        return out_3

if __name__ == '__main__' :  
    '''
    net = LeNet5()
    #print(net)
    input1 = Variable(torch.randn(1,1,28,28))
    input2 = Variable(torch.randn(1,1,16,1))
    output = net(input1 , input2)
    print(output.shape)
    '''
    net = ANN_OTF(11)
    print(net)
    #net = CNN(1, 3)
    input1 = Variable(torch.randn(100,64))
    input2 = Variable(torch.randn(100,96))
    output = net(input1, input2)
    print(output.size())
    '''
    g = make_dot(output)  
    g.view()  
  
    params = list(net.parameters())  
    k = 0  
    for i in params:  
        l = 1  
        for j in i.size():  
            l *= j    
        k = k + l  
    print("总参数数量和：" + str(k))
    '''
    