import resnext
import torch

def gen_random_preprocess(num_perspective=6, num_tilt=3):
    perspective = [resnext.preprocess_random_perspective for i in range(0, num_perspective)]
    tilt = [resnext.preprocess_tilt for i in range(0, num_tilt)]
    flip = [resnext.preprocess_flip_horizontal]
    return perspective + tilt + flip

# calculation performed using method outlined in
# Classification Confidence Estimation with Test-Time Data-Augmentation
# Yuval Bahat, Gregory Shaknarovich
# June 30, 2020
# arXiv: https://arxiv.org/abs/2006.16705
# Basically, just averaging softmax outputs
def calc_confidence(fname):
    with open("imagenet_classes.txt", "r") as f:
        num_categories = len([s.strip() for s in f.readlines()])

    softmax_avg = torch.zeros(num_categories)
    preprocess = gen_random_preprocess()

    for p in preprocess:
        softmax_avg = softmax_avg.add(resnext.resnext_run(fname, preprocess=p))

    return softmax_avg.div(len(preprocess))

def calc_confidence_idx(fname, cat_idx):
    return calc_confidence(fname)[cat_idx].tolist()

def test():
    fname = "testimg/website.jpg"
    prediction = resnext.resnext_classify(fname)
    (certainty, img_classification) = (calc_confidence_idx(fname, prediction[2][0]), prediction[0][0])
    print(certainty)
    print(img_classification)

if __name__ == '__main__':
    test()