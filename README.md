# Earth's Pulse, City's Breath: Enhancing Urban Land Cover Classification through SAR and Sentinel-2 Data Fusion  

## Authors  
Marc Crampe, Emilio Picard  
EuroMoonMars, Terraprisma, IPSA, ENS-Paris-Saclay  

## Abstract  
Urban areas face unique challenges in monitoring and managing land cover changes, particularly due to cloud cover that affects optical imagery. This project presents an innovative approach to urban land cover classification by fusing Sentinel-1 Synthetic Aperture Radar (SAR) data with Sentinel-2 optical data.  

The objective is to develop an automated algorithm capable of monthly monitoring of urban land cover changes. By enhancing classification accuracy, the project aims to provide critical insights into urban growth and environmental resilience.  

A merged dataset of **20 bands** was created using:  
- Sentinel-1: VV, VH, VV/VH (from 4 images).  
- Sentinel-2: Spectral indices (NDVI, SAVI, BAI, NDWI) and original bands (B2, B3, B4, B8).  

Several machine learning models (SVM, KNN, RF, NN) and advanced methods (CNN, SAM2) were evaluated to classify urban features accurately. Results demonstrate the potential of SAR and optical data fusion to overcome the limitations of cloud cover, providing a robust framework for urban land cover monitoring.  

## Case Study: Rouen, France  
Rouen, located in northern France, offers a diverse urban landscape ideal for studying land cover classification.  
### Key Features:  
- **Population:** ~111,000 in the city center, over 500,000 in the metropolitan area.  
- **Challenges:** Urban growth, flood management (Seine River), and environmental remediation.  
- **Focus:** Monitoring urban development, green spaces, and strategies for reducing environmental footprints.  

## Methodology  

### Data Fusion  
1. **Sentinel-1 SAR Data:** Resistant to cloud cover, provides structural information.  
2. **Sentinel-2 Optical Data:** Supplies spectral indices for vegetation health, soil, water bodies, and burned areas.  

### Merged Image Bands:  
The final dataset consists of 20 bands:  
`[VV_1, VV_2, VV_3, VV_4, VH_1, VH_2, VH_3, VH_4, VV/VH_1, VV/VH_2, VV/VH_3, VV/VH_4, NDVI, SAVI, BAI, NDWI, B2, B3, B4, B8]`  

This comprehensive dataset combines structural and spectral information for enhanced classification precision.  

### Models Evaluated:  
- **Pixel-based Machine Learning:** Random Forest (RF), K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Decision Trees (DT).  
- **Deep Learning:** Convolutional Neural Networks (CNN).  
- **SAM2 Model:** Advanced classification technique for complex landscapes.  

### Performance Results:  
- **Random Forest:** Achieved **99.81% accuracy** with 5-fold cross-validation.  
- **CNN:** Showed improvements in spatial precision, reducing misclassification in mixed land cover types.  

### Visualization:  
- Confusion matrices were generated for each model to analyze misclassification rates.  
- Classification images provide detailed insights into land cover distribution.  

## Next Steps  
To enhance classification further, we aim to:  
1. Test additional classifiers to address limitations in complex urban environments.  
2. Improve CNN architecture for better spatial feature learning.  
3. Automate the algorithm for large-scale, monthly urban monitoring.  

## Conclusion  
This project highlights the power of SAR and optical data fusion to overcome cloud cover limitations, enabling precise urban land cover monitoring. The findings provide a robust framework for supporting urban planning, environmental management, and resilience strategies.  

## Acknowledgments  
This work was conducted as part of the collaboration between EuroMoonMars, Terraprisma, and ENS-Paris-Saclay.  
