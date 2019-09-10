# GAN
YBIGTA 신입기수 프로젝트로 진행한 GAN Project입니다. 기본적인 GAN의 구조를 이해하고 좀 더 심화된 모델인 DCGAN, Conditional GAN, Disco GAN, Pix2Pix까지 구현해보았습니다.



## 0. GAN의 Idea
GAN(Generative Adversarial Network)은 두개의 네트워크로 구성된 심층 신경망 구조입니다. 두 개의 네트워크와 서로 대립하는 구조이기 때문에 적대적(Adversarial)이라는 단어가 이름에 들어가있습니다. GAN은 어떠한 분포의 데이터도 학습하여 fake image를 생성할 수 있는 아주 강력한 모델입니다. 학습된 생성 모델은 육안으로 진짜인지 가짜인지 구분할 수 없을 정도로 진짜 같은 fake image를 만들어낼 수 있습니다.

![Alt text](/imgs/GANstructure.png) 

GAN을 이해하기 위해서는 Generator와 Discriminator의 관계를 이해해야합니다. 예를 들어 MNIST 데이터로 숫자이미지를 생성한다고 생각해봅시다. Discriminator의 목적은 실제 MNIST 데이터를 입력했을 때 그것이 진짜인지 가짜인지 판별하는 것입니다. Discriminator는 두 가지 방법으로 학습이 진행됩니다. 하나는 real data를 입력해서 해당 데이터를 진짜로 분류하도록 학습시키는 것입니다. 한편 Generator는 임의의 벡터(Random noise)에 대해 완전히 새로운 fake image를 생성합니다. 그리고 이 fake image를 Discriminator에 입력해서 가짜로 분류하도록 학습시키는 것이 두번째 학습입니다. 

 이러한 학습과정 후에는 학습된 Discriminator를 속이는 방향으로 Generator를 학습시켜야 합니다. 이번에는 fake image를 진짜라고 분류할 때까지 네트워크의 weight값을 조정하여, 진짜와 매우 유사한 데이터를 만들어 내도록 Generator를 학습시키는 것입니다.
 
 
 
## 1. DCGAN
### 1_1. DCGAN의 구조
2014년 처음 발표된 original GAN은 MNIST 같은 비교적 단순한 이미지는 잘 학습했지만, CIFAR-10 같은 복잡한 이미지에 대해서는 그렇게 좋은 이미지를 생성하지 못했습니다. Alec Radford 등이 2015년에 발표한 DCGAN(Deep Convolutional GAN)은 기존의 GAN에 CNN을 적용하여 네트워크 구조를 발전시켰고 최초로 고화질 영상을 생성했습니다. 


* UnSupervised Representation Learning with Deep Convolutional GANs(paper)

     <https://arxiv.org/pdf/1511.06434.pdf>


연구진들은 다양한 실험을 통해 최적의 결과를 나타내는 모델의 구조를 알아냈고, 그 내용은 다음과 같습니다.

1. Max pooling layer를 없애고 strided convolution을 통해 feature map의 크기를 조절한다.
2. Batch Normalization을 적용한다.
3. Fully connected hidden layer를 제거한다.
4. Generator의 마지막 활성함수로 Tanh를 사용하고, 나머지는 ReLU를 사용한다.
5. Discriminator의 활성함수로 LeakyReLU를 사용한다.

![Alt text](/imgs/DCGANstructure.png)

Generator는 latent variable으로부터 64×64 크기의 최종 이미지를 출력해야합니다. Convolution layer로 넘어간 출력은 feature map의 크기를 키워야하기 때문에 fractionally strided convolution이 필요합니다. 일반적으로 1 이상의 stride를 사용하면 pooling과 마찬가지로 출력의 크기를 줄이는 효과를 얻을 수 있습니다. 

또한 연구진들은 입력데이터로 사용하는 random noise인 z값을 살짝 바꾸면 생성되는 이미지가 이에 대응하여 변하게 되는 vector arithmetic의 개념을 찾아냈습니다.
