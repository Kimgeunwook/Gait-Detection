

# Brief Introduction Video
<a href="https://www.youtube.com/watch?v=nBtSBsJISKw">프로젝트 소개 영상</a>


# 목차
### 1. 프로젝트 소개 
### 2. 시스템 구성도  
### 3. 기술적 문제 해결  
<br/>
<br/>
<br/><br/>

# 1. 프로젝트 소개    
***사람이 걷는 영상에서 얻은 실루엣을 바탕으로 re-id 신경망에 입력하여 어떠한 사람과 가장 유사한지 찾아내는 것을 목표로 한다.***
<br/>
<br/>
<br/>
<br/>
          

# 2.시스템 구성도  
<img src="https://user-images.githubusercontent.com/48399897/136685593-79cda924-e855-46b1-aa53-64a7a4c575f4.PNG" width="100%" height="100%"  >  
<br/>  
<br/>  
<br/>  
<br/>  

# 3. 기술적 문제 해결
<details markdown="1">   
<summary> 1. re-id net 신경망 구조 개선 (자세히 보기)</summary>  

##  한정된 후보에서만 비교하던 분류 모델에서 확장 가능한 re-id 신경망 구조로 변경.    

<p align="center"><img src="https://user-images.githubusercontent.com/48399897/137710480-9110f5e3-4680-42f1-8997-5f0201a4a179.PNG" width="100%" height="100%"  ></p>
<p align="center"><img src="https://user-images.githubusercontent.com/48399897/137710473-1eddb1c5-de35-4a9f-aa96-10aee1cbd075.PNG" width="100%" height="100%"  ></p>
</details> 
<br/>  
<br/>  
<details markdown="1">   
<summary> 2. 사람 실루엣 노이즈 개선 (자세히 보기)</summary>  

##  Mask-Rcnn을 사용하여 노이즈 제거 및 실루엣 화질 개선.    

<p align="center"><img src="https://user-images.githubusercontent.com/48399897/137713461-be364fe7-f9f2-4552-a800-0849867ab334.PNG" width="100%" height="100%"  ></p>
<p align="center"><img src="https://user-images.githubusercontent.com/48399897/137712551-146a459b-bb1f-4604-a687-7b56be865428.PNG" width="100%" height="100%"  ></p>
<p align="center"><img src="https://user-images.githubusercontent.com/48399897/137713817-cef44614-ab17-4960-8ad3-d5c0d63d2c17.PNG" width="100%" height="100%"  ></p>
<br/>  
<br/>  
<p align="center"><img src="https://user-images.githubusercontent.com/48399897/137817989-17c049ed-5d7d-4c44-990d-3c83867c05f1.PNG" width="100%" height="100%"  ></p>
</details> 
<br/>  
<br/>  
<br/>  
<br/>  


