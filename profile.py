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


def dummy_test():
	model_original = ModifiedVGG16Model().cuda()
	#for param in model_original.features.parameters():
	#	param.requires_grad = True
	model_original.eval()
	summary(model_original, input_size=(3,224,224))
	print("\n\n__________________________________________________\n\n")

	state_dict = model_original.state_dict()

	model_pruned = PrunedVGGModel(state_dict = state_dict).eval()
	summary(model_pruned, input_size=(3,224,224))


model_original = torch.load("model")
for param in model_original.features.parameters():
	param.requires_grad = True
summary(model_original.features, input_size=(3,224,224))
test(model_original)

print("\n\n__________________________________________________\n\n")

model_pruned = PrunedVGGModel(state_dict=torch.load("model_prunned")).cuda()
summary(model_pruned.features, input_size=(3,224,224))
test(model_pruned)


#dummy_test()

