from torch import nn


class branch_fusion(nn.Module):
    def __init__(self, channel, reduction=16):
        super(branch_fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Linear(channel, channel // reduction, bias=False)
        self.relu=nn.ReLU(inplace=True)
        self.fc2=nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid=nn.Sigmoid()


    def forward(self, x): 
        b, c, _, _ = x.size()        
        y1=self.fc1(x)
        self.fc1_feature=y1
        y2=self.relu(y1)  
        self.relu_feature=y2
        y3=self.fc2(y2)   
        self.fc2_feature=y3
        y4=self.sigmoid(y3).view(b, c, 1, 1)
        self.sigmoid_feature=y4
        y5=x * y4.expand_as(x)
        self.feature=y5
        return y5
    

    
