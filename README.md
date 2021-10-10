# Gait-Based re-identification(In Progress)
Human Re-identification Based on Gait Analysis(GEI)

# Brief Introduction Video
<a href="https://www.youtube.com/watch?v=nBtSBsJISKw">프로젝트 소개 영상</a>


# 목차
### 1. 프로젝트 소개 
### 2. 시스템 구성도  
### 3. re-id network  
### 4. 정확도 분석
<br/>
<br/>
<br/><br/>

# 1. 프로젝트 소개    
영상에서 얻은 사람의 실루엣을 바탕으로 신경망에 입력하여 어떠한 사람과 가장 유사한지를 식별하는 것을 목표로 한다.
<br/>
<br/>
<br/>
<br/>
          

# 2.시스템 구성도  
<img src="https://user-images.githubusercontent.com/48399897/136685593-79cda924-e855-46b1-aa53-64a7a4c575f4.PNG" width="100%" height="100%"  >  
<br/>
<br/>

# 3. re-id network  
<div> <center><img src="https://user-images.githubusercontent.com/48522169/81587239-b9b0e200-93f1-11ea-9e88-425cd1339417.png" width="50%" height="40%" title="reidNet " alt="실행1"> </img></div>
<br/>
<br/>

# 4. 정확도 분석   

***1. 각도별 정확성***    
<div>
<img src="https://user-images.githubusercontent.com/48522169/81587086-7eaeae80-93f1-11ea-800f-3ffc732467b1.png" width="35%" height="35%" title="matrix" alt="실행1">     </img>  
</div>    
</br>
</br>
***2. 실제 정확성***    
<div>
<img src="https://user-images.githubusercontent.com/48522169/81588074-ec0f0f00-93f2-11ea-8414-a2b889793510.png" width="35%" height="35%" title="" alt="실행1">     </img>  
</div>    
</br>
</br>

## Citation
```
@InProceedings{BMSengupta20,
  title={Background Matting: The World is Your Green Screen},
  author = {Soumyadip Sengupta and Vivek Jayaram and Brian Curless and Steve Seitz and Ira Kemelmacher-Shlizerman},
  booktitle={Computer Vision and Pattern Regognition (CVPR)},
  year={2020}
}

@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}

@INPROCEEDINGS{7299016,
  author={E. {Ahmed} and M. {Jones} and T. K. {Marks}},
  booktitle={2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={An improved deep learning architecture for person re-identification}, 
  year={2015},
  volume={},
  number={},
  pages={3908-3916},}
```
