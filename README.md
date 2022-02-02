# DilatedCRF
Pytorch implementation for fully-learnable DilatedCRF. 
***********************************
If you find my work helpful, please consider our paper:
```
@article{Mo2022dilatedcrf,
    title={Dilated Continuous Random Field for Semantic Segmentation},  
    author={Xi Mo, Xiangyu Chen, Cuncong Zhong, Rui Li, Kaidong Li, Sajid Usman},
    booktitle={IEEE International Conference on Robotics and Automation}, 
    year={2022}  
}
```
***********************************
## Easy Setup
Please install these required packages by official guidance:

`python >= 3.6`   
`pytorch >= 1.0.0`  
`torchvision`   
`pillow`   
`numpy`

## How to Use
### 1. Prepare dataset
* Dowload `suction-based-grasping-dataset.zip` (1.6GB) [[link](https://vision.princeton.edu/projects/2017/arc/)].
Please cite relevant paper:
```
@article{zeng2018robotic, 
    title={Robotic Pick-and-Place of Novel Objects in Clutter with Multi-Affordance Grasping and Cross-Domain Image Matching},  
    author={Zeng, Andy and Song, Shuran and Yu, Kuan-Ting and Donlon, Elliott and Hogan, Francois Robert and Bauza, Maria and Ma, Daolin and Taylor, Orion and Liu,     Melody and Romo, Eudald and Fazeli, Nima and Alet, Ferran and Dafle, Nikhil Chavan and Holladay, Rachel and Morona, Isabella and Nair, Prem Qu and Green, Druck and Taylor, Ian and Liu, Weber and Funkhouser, Thomas and Rodriguez, Alberto},  
    booktitle={Proceedings of the IEEE International Conference on Robotics and Automation}, 
    year={2018}  
}
```
* Train your own semantic segmentation classifers on the suction dataset, generate training samples and test samples for DilatedCRF. You can also download my 
training set and test set (872MB) [[link](https://drive.google.com/file/d/1EhXTVPmgE4mg4ImnKEsd72St2O_zHE2d/view?usp=sharing)], extract the default folder `dataset` to the main directory.   
**NOTE: Customized training and test samples must be organized the same as the default dataset format.**

### 2. Train network
* If you want to customize training process, modify `utils/configuration.py` parameters according to its instructions.
* Train DilatedCRF use default `dataset` folder, or customized dataset path by `-d` argument.   
**NOTE: checkpoints will be written to the default folder `checkpoint`.**
    ```
    python DialatedCRF.py -train
    ```  
    or restore training using the lattest `.pt` file stored in default folder `checkpoint`:
    ```
    python DialatedCRF.py -train -r
    ```
    or you may want to use specified checkpoint:
    ```
    python DialatedCRF.py -train -r -c path/to/your/ckpt
    ```
    Note that checkpoint file must match the parameter `"SCALE"` specified in `utils/configuration.py`. To specify customized dataset folder, use:   
    
    ```
    python RGANet.py -train -d your/dataset/path
    ```
    
### 3. Validation
* Complete dataset folder mentioned above and a valid checkpoint are required. You can download my checkpoint for `"SCALE" = 0.25` (42.4MB) 
[[link](https://drive.google.com/file/d/1fMhrxR9mIx3MthqzI3Z94mxjoOj8knHk/view?usp=sharing)], be sure to adjust corresponding configurations beforehand. Then run:

    ```
    python DialatedCRF.py -v
    ```
    or you may specify dataset folder by `-d`:
    ```
    python DialatedCRF.py -v -d your/path/to/dataset/folder
    ```
* Final results will be written to folder `results`. Metrics including `Jaccard`, `F1-score`, `accuracy`, etc., will be gathered as `evaluation.txt` in the folder `results/evaluation`
********************************** 
Contributed by Xi Mo,   
License: Apache 2.0
