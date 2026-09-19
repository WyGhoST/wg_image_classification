
import torch

def softmax(logits, dim=1):
    #Tìm giá trị logits max sau đó trừ ra để tránh gây tràn số
    max_values = logits.max(dim=dim, keepdim=True).values
    shifted_logits = logits - max_values
    
    #Lấy e^ 
    exp_scores = torch.exp(shifted_logits)
    
    sum_exp_scores = exp_scores.sum(dim=dim, keepdim = True)
    
    #Tiến hành chuẩn hóa
    prob = exp_scores/sum_exp_scores
    
    return prob
    

class SoftmaxClassifier:
    def __init__(self, input_size= 784, num_classes = 10):
        
       self.W = torch.randn(
        input_size,
        num_classes
        ) * 0.01
       
       self.b = torch.zeros(num_classes)
 
       
    def forward(self, images):
        batch_size= images.shape[0]
        x = images.reshape(batch_size, -1)
        logits = x @ self.W + self.b
        return logits
    
    def __call__(self, images):
        return self.forward(images)
    
    def predict_prob(self, images):
        logits = self.forward(images)
        prob = softmax(logits, dim=1)
        return prob
    
    def predict(self, images):
        logits = self.forward(images)
        
        prediction = torch.argmax(logits, dim=1)
        
        return prediction

        