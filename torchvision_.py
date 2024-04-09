import torchvision

train_set=torchvision.datasets.CIFAR10(root='set01',train=True,download=True)
test_set=torchvision.datasets.CIFAR10(root='set01',train=False,download=True)
print(test_set.classes)

img,label=train_set[0]
print(img)
print(label)
img.show()