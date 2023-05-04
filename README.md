# PedestrianDetection
codes for pedestrian detection
## CSP
* Origin repo: https://github.com/polariseee/CSP-pedestrian-pytorch . Codes have been made some changes in this repo
* My best MR^-2 is **5.59%**, a little better from the author's(5.84%)
* follow the origin repo to configure, and follow the UseIntro.txt to use my code.
## F2DNet
refer to the Readme.md in file F2DNet
## Pedestron
refer to the Readme.md in file Pedestron
## My method
The structure of CSP:  
![image](https://user-images.githubusercontent.com/94534877/235969537-50fa8d3c-50c7-48b3-9664-b1a19fef68f3.png)
### CSP_HeadCenter_3×3
* add a head branch(**head cente**r) with [conv 1×1×256] after "conv 3×3×256"
### CSP_Head_FeatureMap
* add a head branch(**head center + head scale + offset**) with [conv 3×3×256 --> 3* conv 1×1×256]  after feature map
### CSP_Head_3×3
* add a head branch(**head center + head scale + offset**) with [3* conv 1×1×256]  after "conv 3×3×256"
## My results
* training and testing different methods using different loss weights
* best result is **5.42%MR^-2** with **CSP_HeadCenter_3×3** and the loss weight of head center is **0.005**.  (**a 0.17% improvement compared with the CSP code!**)
