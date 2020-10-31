# TinyImageNet-Pytorch-MobileNetV2

TinyImageNet에 대한 내용과 접근 방식은 VGGNet을 참고하시면 됩니다.

(https://github.com/kjo26619/TinyImageNet-Pytorch-VGGNet)

# MobileNet V1

MobileNet은 VGGNet에서 모바일 등에서 효율적으로 사용하기 위해서 나온 CNN모델입니다.

기존의 VGGNet은 3x3 Convolution Layer를 반복하면서 네트워크의 깊이 즉, Layer의 수를 증가시켰고 이는 높은 Accuracy를 가지게 됩니다.

하지만 여기서 문제가 발생합니다. 3x3 Convolution Layer와 중간에 섞여있는 Max Pooling Layer가 많은 파라미터의 수와 연산을 가지게 됩니다.

MobileNet은 Depthwise Separable Convolution이라고 하는 연산으로 바꾸는 것입니다.

## Depthwise Separable Convolution

![img1](https://github.com/kjo26619/TinyImageNet-Pytorch-MobileNetV2/blob/main/image/depthwise.png)

Depthwise Separable Convolution은 먼저 3x3 Convloution Layer를 각 채널별로 진행을 합니다.

그림에서 보면 빨간색, 파란색, 초록색을 각각 3x3 Convloution 하는 것입니다.

이 과정을 Depthwise Convloution이라고 합니다.

그리고 각 채널별로 3x3 Convolution을 했으면, 이를 다시 모아서 1x1 Convolution Layer를 진행합니다.

이 과정을 Pointwise Convolution이라고 합니다.

이러한 Depthwise Separable Convolution이 왜 파라미터를 줄일 수 있는지는 기존의 연산과 비교해보면 알 수 있습니다.

기존 연산의 경우에는 Kernel Size * Kernel Size * Input Channel * Output Channel로 파라미터를 가지게 됩니다.

만약 3의 Kernel을 갖는 Convolution Layer가 있고 Input이 128이고 Output이 256이면, 3 * 3 * 128 * 256 = 294,912 입니다.

그러나, Depthwise Convloution은 Kernel Size * Kernel Size * Input Channel 입니다. 

그리고 Pointwise Convolution은 Input Channel * Output Channel 입니다. (1x1 연산이기 때문)

아까와 같이 3의 Kernel에 Input 128, Output 256이면 

Depthwise : 3 * 3 * 128 = 1,152

Pointwise : 128 * 256 = 32,768

둘을 더하는 것이 Depthwise Separable Convolution 이므로

1,152 + 32,768 = 33,920입니다.

약 8배 정도의 파라미터가 줄어들었습니다.

이러한 Depthwise Separable Convolution을 통해서 파라미터를 줄이고 모바일에서도 효율적으로 사용할 수 있는 것이 바로 MobileNet입니다.

# MobileNet V2

MobileNet도 점차 진화하여 새로운 버전이 나오게 됩니다.

MobileNet V2도 여전히 Depthwise Separable Convolution을 사용하지만, V1과는 다르게 ResNet과 유사한 방식으로 사용합니다.

ResNet과 다른점은 바로 Inverted Residual Bottleneck 구조라는 것입니다.

Inverted Residual Bottleneck 구조는 기존의 3x3 Convolution Layer를 하기 전에 Channel을 줄이는 작업으로 1x1 Convolution를 사용한 것과는 반대로

1x1 Convolution을 이용하여 채널의 수를 확장시킵니다. 그리고 3x3 Convolution Layer에서 Depthwise Convolution을 진행합니다.

그리고 다시 1x1 Convolution을 이용하여 Pointwise Convolution을 진행합니다.

여기서 자잘하게 ResNet과 MobileNetV1과는 차이를 보이는데,

먼저 ReLU6를 사용했다는 점입니다.

그리고 Stride가 2 즉, Down Sampling 할 때는 Skip Connection을 사용하지 않습니다.

마지막으로 Pointwise Convolution을 할 때, Activation Function 없이 Linear하게 출력됩니다.
