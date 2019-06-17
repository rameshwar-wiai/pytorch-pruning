from torchsummary import summary
import torch
from finetune import *
from dataset import test_loader

def test(model):
	test_data_loader = test_loader('/tmp/mnist/')
	model.eval()
	correct = 0
	total = 0

	for i, (batch, label) in tqdm(enumerate(test_data_loader)):
		batch = batch.cuda()
		output = model(Variable(batch))
		pred = output.data.max(1)[1]
		correct += pred.cpu().eq(label).sum()
		total += label.size(0)
	
	print("Accuracy :", float(correct) / total)



model_original = torch.load("model")
for param in model_original.features.parameters():
	param.requires_grad = True
summary(model_original.features, input_size=(3,224,224))
test(model_original)

model_pruned = PrunedVGGModel(state_dict=torch.load("model_prunned")).cuda()
print(torch.load("model_prunned").keys())
summary(model_pruned.features, input_size=(3,224,224))
test(model_pruned)