# GAN
YBIGTA 신입기수 프로젝트로 진행한 GAN Project입니다. 기본적인 GAN의 구조를 이해하고 좀 더 심화된 모델인 DCGAN, Conditional GAN, Disco GAN, Pix2Pix까지 구현해보았습니다.


# 0. GAN  
GAN(Generative Adversarial Network)은 두개의 네트워크로 구성된 심층 신경망 구조입니다. 두 개의 네트워크와 서로 대립하는 구조이기 때문에 적대적(Adversarial)이라는 단어가 이름에 들어가있습니다. GAN은 어떠한 분포의 데이터도 학습하여 fake image를 생성할 수 있는 아주 강력한 모델입니다.      
GAN을 이해하기 위해서는 Generator와 Discriminator의 관계를 이해해야합니다.
