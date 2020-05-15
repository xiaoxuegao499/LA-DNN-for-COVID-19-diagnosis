# LA-DNN for COVID-19 diagnosis

Online COVID-19 diagnosis with chest CT images: Lesion-attention deep neural networks

## Background

Chest (computed tomography) CT scanning is one of the most important technologies for COVID-19 diagnosis in the current clinical practice, which motivates more concerted efforts in developing AI-based diagnostic tools to alleviate the enormous burden on the medical system. We develop a lesion-attention deep neural network (LA-DNN) to predict COVID-19 positive or negative with a richly annotated chest CT image dataset. The CT image dataset contains 746 public chest CT images of COVID-19 patients collected from over 760 preprints, and the data annotations are accompanied with the textual radiology reports. We extract two types of important information from these annotations: One is the flag of whether an image indicates a positive or negative case of COVID-19, and the other is the description of five lesions on the CT images associated with the positive cases. The proposed data-driven LA-DNN model focuses on the primary task of binary classification for COVID-19 diagnosis, while an auxiliary multi-label learning task is implemented simultaneously to draw the model's attention to the five lesions of COVID-19 during the training. The joint task learning process makes it a highly sample-efficient deep model that can learn COVID-19 radiology features effectively with very limited samples. The experimental results show that the area under the curve (AUC) and sensitivity (recall) for the diagnosis of COVID-19 patients are 91.2% and 85.7% respectively, which reach the clinical standards for practical use. An online system has been developed for fast online diagnoses using CT images at the web address https://www.covidct.cn/.

## Data

* We used this public datasets: **"COVID-CT-Dataset: a CT scan dataset about COVID-19."** arXiv, 2020. <br>
  The following link can help you get a detailed description about the dataset.<br>
  arXiv: https://arxiv.org/abs/2003.13865 <br>
  dataset: https://github.com/UCSD-AI4H/COVID-CT<br>
* In order to further optimize the performance of our model, we are still continuously collecting new CT images, including COVID-19 and NonCOVID-19.


## Citation

Our method and results are described in [Online COVID-19 diagnosis with chest CT images: Lesion-attention deep neural networks](https://www.medrxiv.org/content/10.1101/2020.05.11.20097907v1).<br>
Please cite our paper if you find the work useful:

    @article {Liu2020.05.11.20097907,
      author = {Liu, Bin and Gao, Xiaoxue and He, Mengshuang and Lv, Fengmao and Yin, Guosheng},
      title = {Online COVID-19 diagnosis with chest CT images: Lesion-attention deep neural networks},
      year = {2020},
      doi = {10.1101/2020.05.11.20097907},
      publisher = {Cold Spring Harbor Laboratory Press},
      URL = {https://www.medrxiv.org/content/early/2020/05/14/2020.05.11.20097907},
      eprint = {https://www.medrxiv.org/content/early/2020/05/14/2020.05.11.20097907.full.pdf},
      journal = {medRxiv},
    }
    
## 

