# TinyImageNet-Pytorch-MobileNetV2

TinyImageNet에 대한 내용과 접근 방식은 VGGNet을 참고하시면 됩니다.

(https://github.com/kjo26619/TinyImageNet-Pytorch-VGGNet)

# MobileNet

MobileNet은 VGGNet에서 모바일 등에서 효율적으로 사용하기 위해서 나온 CNN모델입니다.

기존의 VGGNet은 3x3 Convolution Layer를 반복하면서 네트워크의 깊이 즉, Layer의 수를 증가시켰고 이는 높은 Accuracy를 가지게 됩니다.

하지만 여기서 문제가 발생합니다. 3x3 Convolution Layer와 중간에 섞여있는 Max Pooling Layer가 많은 파라미터의 수와 연산을 가지게 됩니다.

MobileNet은 Depthwise Separable Convolution이라고 하는 연산으로 바꾸는 것입니다.

Depthwise Separable Convolution은 
