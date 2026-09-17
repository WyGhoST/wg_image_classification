
import torch

def softmax(logits, dim=1):
    #Tìm giá trị logits max sau đó trừ ra để tránh gây tràn số
    max_values = logits.max(dim=dim, kdim='True').values
    shifted_logits = logits - max_values
    
    #Lấy e^ 
    exp_scores = torch.exp(shifted_logits)
    
    sum_exp_scores = exp_scores.sum(dim=dim, kdim = 'True')
    
    #Tiến hành chuẩn hóa
    prob = exp_scores/sum_exp_scores
    
    return prob
    

class SoftmaxClassifier:
    def __init__(self, input_size= 784, num_classes = 10):
        
       self.W = torch.randn(
           input_size=input_size,
           num_classes=num_classes
       )*0.01
       
       self.b = torch.zeros(num_classes)
       
       self.W.requires_grad_(True)
       self.b.requires_grad_(True)
       
    def forward(self, images):
        batch_size= images.shape[0]
        x = images.reshape
        