# Identifying Eye Diseases with Residual Attention Networks

** The progress thus far are preliminary results. I am still in the process of rigorous validation of the efficiency of this deep learning model **

## Eye Diseases

- The leading causes of blindness
    - Age-related macular degeneration: Almost 10mill individuals suffer in the US
    - Diabetic Retinopathy: Nearly 750K individuals aged 40+ suffer
- Diseases likely to increase due to the aging population and global diabetes epidemic
- Optical Coherence Tomography (OCT) Imaging special way to visualize individual retina layers so that specialists can analyze the state of the retina
- With the OCT images, creating a diagnostic tool based on Residual Attention Networks (RAN) for the screening of patients with common treatable blinding retinal diseases may ultimately aid in expediting the diagnosis and referral of treatable conditions 

## Optical Coherence Tomography (OCT) Imaging
- OCT imaging is the standard retinal imaging tool for the diagnosis and treatment of some of the leading causes of blindness
    - approx. 30 million OCT scans are performed each year world-wide
- Provides a clear visualization of individual retinal layers that would be impossible with clinical examination by the human eye 
- OCT uses light waves to capture distinctive layers of the retina in vivo 

## Residual Attention Networks

- **Paper Title**: “Residual Attention Network for Image Classification”
- **Authors**: Wang, Fei and Jiang, Mengqing and Qian, Chen and Yang, Shuo and Li, Cheng and Zhang, Honggang and Wang, Xiaogang and Tang, Xiaoou
- Convolutional Neural Network with Attention Mechanism and Residual Units
- Built by stacking Attention Modules, which generate attention-aware features. 
- These attention-aware features from different modules change adaptively as layers going deeper and brings more discriminative feature representation
- Inside each Attention Module, bottom-up top-down feedforward structure is used to unfold the feedforward and feedback attention process into a single feedforward process. 
- Implements attention residual learning to train very deep Residual Attention Networks

## Interesting Properties About Architecture

- Increasing Attention Modules leads to consistent performance improvement
    - With multiple Attention Modules, different types of attention are able to be captured
- Attention Residual Learning: naively stacking Attention Modules directly would lead to the obvious performance drop. 
    - Proposes attention residual learning mechanism - optimizes very deep Residual Attention Network with hundreds of layers. 
- Bottom-Up Top-Down Feedforward Attention Structure as part of Attention   - Module to add soft weights on features. 
- Mimics bottom-up fast feedforward process and top-down attention feedback in a single feedforward process 
    - Allows an end-to-end trainable network

## Attention Modules

- Main component of network
- Divided into two branches (components)
    - Mask branch
    - Trunk branch
- Includes Residual Units within module and branches

## Residual Units

- Inspired by Residual Neural Network (ResNet)
- Utilizes skip-connections to jump over 2–3 layers with nonlinearities (e.g. in ReLU CNNs) and batch normalizations
- Motivation for skipping: to avoid the vanishing gradients and degradation problem
- Vanishing Gradient: when the gradient becomes vanishingly small due to the deepness of the network
    - Thus preventing the weights from changing its value and stopping the network from further training
- The degradation problem occurs as network depth increases. As additional layers are added to the network, its accuracy decreases despite being capable of gathering all of the intricacies of our data. 

## Branches

- **Trunk Branch** performs feature processing with Residual Units
- **Mask Branch** uses bottom-up top-down structure softly weight output features with the goal of improving trunk branch features
- Bottom-up Step: collects global information of the whole image by downsampling (i.e. max pooling) the image
- Top-down Step: combines global information with original feature maps by upsampling (i.e. bilinear interpolation) to keep the output size the same as the input feature map. 
- Two consecutive 1 × 1 convolution layers follows 
- Finally, a sigmoid layer normalizes the output range to [0, 1] 

## Attention Residual Learning

- Naive stacking Attention Modules leads to a performance drop
    - Dot production with mask range from zero to one repeatedly will degrade the value of features in deep layers. 
    - Soft mask can potentially break good property of trunk branch, for example, the identical mapping of Residual Unit.
- Attention Residual Learning (ARL) eases the above problems. 
- Similar to ideas in residual learning, if soft mask unit can be constructed as identical mapping,
- Modify output H of Attention Module as
    - Original Residual Learning Formula: H(x) = F(x) * x
    - ARL Formula: H(x) = F(x) * (1+M(x))
- The key difference lies in the mask branches M(x). They work as feature selectors which enhance good features and suppress noises from trunk features.

## Retina OCT Dataset 

- Before training, each image went through a tiered grading system of trained graders of increasing expertise for verification and correction of image labels. 
    - First tier of graders: undergraduate and medical students who had taken and passed an OCT interpretation course review
    - Second tier of graders: four ophthalmologists who independently graded each image that had passed the first tier
    - Third tier of graders: two senior independent retinal specialists, each with over 20 years of clinical retina experience, verified the true labels for each image. 
- Resulted in 84,484 images
- 4 Categories 
    - Choroidal Neovascularization – 37,455
    - Diabetic Macular Edema – 11,598
    - Drusen -  8,866
    - Normal – 26,565

 ## Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning
- Paper that inspired this project
- Authors : Daniel S. Kermany, Michael Goldbaum, Wenjia Cai, ..., M. Anthony Lewis, Huimin Xia, Kang Zhang
- Utilized a pre-trained network which demonstrated competitive performance of OCT image analysis that was comparable to that of human experts with significant clinical experience with retinal diseases. 
- Obtained immaculate results
- Read paper here: **[Paper Link](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)**

## Acknowledgements
- For my preliminary results: **[Notebook Link](https://github.com/deontaepharr/Eye-Disease-Classification-With-Residual-Attention-Networks/blob/master/Notebooks/Eye%20Disorder%20Classification%20with%20Residual%20Attention%20Network.ipynb)**
- Residual Attention Network for Image Classification: **[Paper Link](https://arxiv.org/abs/1704.06904)**
- Deep Residual Learning for Image Recognition:  **[Paper Link](https://arxiv.org/abs/1512.03385)**
- Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning: **[Source Link](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)**
- Data and Code: **[Source Link](https://data.mendeley.com/datasets/rscbjbr9sj/2)**
- Kaggle Source: **[Source Link](https://www.kaggle.com/paultimothymooney/kermany2018)**


