import torch

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
model.eval()

from PIL import Image
from torchvision import transforms

preprocess_default = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

preprocess_random_perspective = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

preprocess_flip_horizontal = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

preprocess_tilt = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def resnext_run(filename, preprocess=preprocess_default):
    # sample execution (requires torchvision)
    input_image = Image.open(filename).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    #if torch.cuda.is_available():
    #    input_batch = input_batch.to('cuda')
    #    model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    return probabilities

def resnext_classify(filename, printout=False):

    probabilities = resnext_run(filename)

    print(probabilities.size())

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    if printout:
        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())

    topfive_cats  = []
    topfive_probs = []

    for i in range(top5_prob.size(0)):
        topfive_cats.append(categories[top5_catid[i]])
        topfive_probs.append(top5_prob[i].item())

    return [topfive_cats, topfive_probs, top5_catid]

def test():
    print(resnext_classify("testimg/dog.jpg", printout=True))

if __name__ == '__main__':
    test()