
Earth's pulse, city's breath: Enhancing urban land cover through SAR and Sentinel-2 data fusion 
Marc CRAMPE, Emilio PICARD
EuroMoonMars, Terraprisma, IPSA, ENS-Paris-Saclay


Abstrast :
Urban areas face unique challenges in monitoring and managing land cover changes, particularly due to cloud cover that affects optical imagery. This study presents an innovative approach to urban land cover classification by fusing Sentinel-1 Synthetic Aperture Radar (SAR) data with Sentinel-2 optical data. We aim to develop an automated algorithm capable of providing monthly monitoring of urban land cover changes, enhancing classification accuracy and offering critical insights into urban growth and environmental resilience.
Utilizing a combination of 4 images from Sentinel-1 (VV, VH, VV/VH) and 5 key spectral indices derived from Sentinel-2 (NDVI, SAVI, BAI, NDWI, and alternative BAI), we created a comprehensive merged dataset of 20 bands. This dataset serves as the foundation for our classification models, including Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Random Forest (RF), and Neural Networks (NN), which are evaluated for their performance in accurately classifying various urban features.
Additionally, we explore Convolutional Neural Networks (CNN) and the SAM2 model for advanced classification techniques. Our results demonstrate the potential of SAR and optical data fusion to overcome the limitations of cloud cover, providing a robust framework for automated urban land cover monitoring. This research contributes to a better understanding of urban dynamics and supports effective environmental management strategies.

Introduction
In recent years, approximately 90% of the world's mapping data has been generated, underscoring the critical importance of leveraging state-of-the-art technologies to create accurate maps that reflect the real challenges facing our planet today. The rapid pace of urbanization has transformed cities, leading to an urgent need for precise tools that can monitor land cover evolution. By comprehensively understanding these changes, we can identify detrimental impacts on the environment while also highlighting effective remediation efforts, ultimately contributing to the development of greener, more sustainable urban landscapes.
However, the utility of optical imagery, particularly from satellites like Sentinel-2, is often compromised by cloud coverage, especially in tropical and coastal regions. This phenomenon results in significant data gaps, which can severely limit the effectiveness of urban monitoring tools. For instance, in Rouen, France, persistent cloud cover restricts the availability of usable optical images to only ten per year, presenting a substantial challenge for consistent urban analysis.
To address these limitations, this study focuses on the exploitation of classification models on fused Sentinel-1 Synthetic Aperture Radar (SAR) and Sentinel-2 optical data. By employing data fusion techniques to mitigate the impacts of cloud coverage, we aim to enable more reliable monthly monitoring of urban land cover changes. The primary objective of this research is to evaluate and compare various machine learning and deep learning models—including Random Forest, Support Vector Machines (SVM), and Convolutional Neural Networks (CNNs)—to identify the most effective methodologies for achieving accurate and robust land cover classification in urban environments.

 
Diagram of algorithm implementaion
 
 
Why Rouen?
Rouen, a historic city in northern France, presents a diverse urban landscape with a mixture of commercial, residential, and industrial zones, as well as surrounding vegetation and water bodies (the Seine River). Its coastal proximity also makes it a relevant choice for studying urban resilience and environmental dynamics, as it faces potential impacts from coastal erosion and urban expansion.

Urban Dynamics and Environmental Challenges
- Population: Approximately 111,000 inhabitants in the city center, with a wider metropolitan area exceeding 500,000 residents.
- Urban Challenges: Urban growth, environmental remediation efforts, and flood management due to its proximity to the river and low-lying coastal areas.
- Environmental Focus: Monitoring urban development, green spaces, and remediation strategies aimed at reducing the environmental footprint of urban sprawl.
 
 Case study: City of Rouen, France

 Application of Data Fusion and Models
For this study, we applied the Sentinel-1 and Sentinel-2 data fusion approach to analyze the land cover changes in Rouen. The various machine learning models (SVM, KNN, RF, NN, CNN, and SAM2) were trained and tested on this region, allowing us to evaluate the accuracy and robustness of each approach in distinguishing between urban features.

We utilize data from both Sentinel-1 (SAR) and Sentinel-2 (optical) to overcome challenges like cloud cover in urban land cover classification.
Sentinel-1 provides radar data (VV, VH, and VV/VH ratio) that is resistant to cloud cover, allowing consistent monitoring of urban structures and vegetation.
Sentinel-2 supplies optical data, from which we calculate 4 key spectral indices specifically designed to distinguish urban features:
- NDVI (Normalized Difference Vegetation Index): Highlights vegetation health.
- SAVI (Soil Adjusted Vegetation Index): Adjusts for soil influence in areas with sparse vegetation.
- BAI (Burned Area Index): Detects bare soil and sparse vegetation.
- NDWI (Normalized Difference Water Index): Identifies water bodies and moisture.
 
The use of these indices allows us to highlight specific features that are not visible to the naked eye by combining different spectral bands to extract precise information. For example, the NDVI, which uses the red and near-infrared bands, detects vegetation health by distinguishing areas with dense vegetation cover from those where vegetation is in decline. Similarly, the SAVI, a modified version of NDVI that accounts for exposed soil, is crucial in environments with sparse vegetation. The BAI, using blue and near-infrared bands, detects areas with bare soil or minimal vegetation, often associated with phenomena like fires or erosion. Lastly, the NDWI, which combines the green and near-infrared bands, identifies water bodies or moisture presence, providing key information for assessing hydrology and flood risks.
When combined with Sentinel-1 radar data, which is resistant to cloud cover, these indices enable more accurate and detailed analysis of urban land cover changes. This fusion of data enhances our ability to monitor and predict environmental transformations in cities, improving decision-making for urban planning and disaster mitigation

 Multispectral Data Visualization of Sentinel-1 and Sentinel-2
 
 
The final merged image consists of 20 bands:
Sentinel-1: 4 images × 3 bands (VV, VH, VV/VH).
Sentinel-2: The 5 special indices plus the original B2, B3, B4, B8 bands.
 
 
Final merged image: 20 bands
 
 
The final merged image consists of 20 bands, formulated as follows:
Merged Image = [VV_1, VV_2, VV_3, VV_4, VH_1, VH_2, VH_3, VH_4, VV/VH_1, VV/VH_2, VV/VH_3, VV/VH_4, NDVI, SAVI, BAI, NDWI, B2, B3, B4, B8].
 
This fusion allows us to capture both structural and spectral information, enhancing the precision of our urban land cover classification.
Pixel based classification
 
Land cover classification involves categorizing areas of land into distinct classes based on observed features. This process is crucial for understanding environmental conditions, managing natural resources, and monitoring changes in land use. In this study, we focus on pixel-based classifiers, which analyze each pixel in satellite imagery independently to determine its class based on spectral signatures.

We evaluated several pixel-based classification models, including Random Forest, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Decision Tree classifiers. These models utilize the spectral information from merged Sentinel-1 and Sentinel-2 data to classify pixels into categories such as vegetation, water, urban areas, forest, and crops.
 
To determine the effectiveness of each classifier, we computed confusion matrices and overall accuracy metrics.
 
 Models’ performance
 
The Random Forest model achieved an impressive accuracy of 99.81%, evaluated through 5-fold cross-validation.

This method enhances the robustness of the model by averaging the performance across multiple iterations, ensuring that the results are reliable and not influenced by a single data split.

 Confusion matrices for: RF, SVM, KNN, DT

 
The confusion matrices revealed the model's ability to accurately distinguish between classes, with minimal misclassification.
 
Here are the four classification outputs obtained after training our models on the same training set. Each model—Random Forest, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Decision Tree—provides insights into their respective performance in land cover classification.
 
 
 
Land cover from the ML Models
 
Upon examining the classification images, we can observe that there are instances of misclassification in specific areas, primarily due to the presence of multiple land cover types. Notably, some models struggle to accurately classify these regions:
 
Area of misclassification
 
KNN and SVM misidentify urban areas as water, indicating a lack of sensitivity in distinguishing between these classes in certain contexts.
The Decision Tree model inaccurately classifies an urban zone as crops, highlighting its tendency to oversimplify the classification boundaries.
These misclassifications reveal that not all models perform equally well in complex landscapes, reinforcing the need for careful selection of classification algorithms based on the specific characteristics of the data.
 
 
Best ML Land Cover
 
The following image illustrates the classification results from the Random Forest model, showcasing the detailed land cover classification across Rouen, France.
Given these observations, we plan to explore additional types of classifiers, particularly image-based methods, to further enhance classification accuracy and address the limitations encountered with the pixel-based classifiers.
CNN classification
 
In this section, we present the results obtained from using a Convolutional Neural Network (CNN) to perform land cover classification. Unlike pixel-based classifiers, CNNs are capable of capturing spatial features and patterns within the image, allowing for a more comprehensive understanding of the landscape. This technique has proven effective in reducing misclassification in areas with mixed land cover types, as the CNN can leverage contextual information to improve classification accuracy.
Below are the classification results using CNN on the same dataset as the pixel-based classifiers. The CNN was trained using the same training samples, but its ability to capture spatial features led to better performance in complex regions.
 
 
CNN Architecture
 
The architecture of the Convolutional Neural Network (CNN) consists of several layers designed to capture spatial features and patterns in the input data effectively. It begins with a 2D convolutional layer (Conv2D) that outputs a feature map with 32 filters, allowing the model to learn complex patterns in the data.

This is followed by a max pooling layer, which reduces the spatial dimensions, thus minimizing the number of parameters and computation in the network. The architecture includes additional dense layers that progressively extract and refine features, leading to the final output layer with five neurons corresponding to the different land cover classes.

The total number of parameters in the model is 132,497, with 44,165 being trainable. This layered approach enables the CNN to learn hierarchical representations of the input data, contributing to its effectiveness in land cover classification. 

 CNN Performance
 
Performance Overview:
Accuracy: 98.01 %
 
While the CNN performs better overall, there are still areas for improvement, especially in regions with subtle differences in land cover types.
 
 
CNN Land Cover of Rouen
 
In conclusion, CNN models offer a robust alternative to pixel-based classification methods, particularly in challenging areas.
However, further refinement and larger datasets could improve performance even more. Future work will explore more advanced deep learning models that can handle both spatial and spectral information.

As the field of machine learning evolves, so does the exploration of innovative models that can handle the complexities of land cover classification more effectively. In this study, we delved into the potential of the Zero Shot Prediction model utilizing the SAM2 framework, which offers a novel approach to land cover classification. Unlike traditional methods that require extensive training datasets, Zero Shot Learning (ZSL) allows the model to make predictions on unseen classes based on prior knowledge. This approach is particularly advantageous in scenarios where labeled data is scarce or difficult to obtain, which is often the case in remote sensing applications.

Preliminary results from applying the Zero Shot Prediction model indicate promising performance, especially in recognizing classes that were not explicitly represented in the training dataset. However, it is essential to note that this method demands significant computational resources and precision in its implementation. The complexity of the model can lead to challenges in real-time applications, especially in urban monitoring scenarios that require timely and efficient processing of large datasets. 
 
 

 
https://res.cloudinary.com/amuze-interactive/video/upload/q_auto/v1727869908/iaf/FE-D9-48-57-BC-F3-5D-D9-AA-DD-AD-FB-5F-E7-AD-59/Video/Enregistrement_de_l_e%CC%81cran_2024-10-02_a%CC%80_13.35.15_h8jcrn.mp4 
 
We recognize that while the SAM2 model presents challenges in terms of resources, it also offers significant opportunities for the future. Future research will focus on optimizing the Zero Shot Prediction model for urban land cover classification, exploring hybrid approaches that combine ZSL with traditional machine learning methods to maximize accuracy and efficiency. By refining this approach, we aim to harness the strengths of both methodologies, ultimately contributing to a more comprehensive understanding of urban dynamics and land cover evolution.


Conclusion:

The integration of Sentinel-1 and Sentinel-2 data demonstrates significant potential for improving urban land cover classification amidst challenges posed by cloud cover. Our findings highlight the effectiveness of Random Forest and CNN models while addressing the limitations of traditional pixel-based methods.
Future work will focus on refining classification algorithms and exploring the applicability of advanced deep learning models that can harness both spatial and spectral data. Additionally, we will make the complete code accessible on GitHub and Streamlit, allowing users to adapt land cover classification for specific case studies.







Acknowledgments

We would like to acknowledge the contributions of the European Space Agency for providing Sentinel data and the collaborative efforts of our research team in the successful execution of this study.

Next Steps & Code Accessibility:

The complete code for all models, including the pixel-based classifiers, CNN, and SAM, will be accessible soon on GitHub and Streamlit. The code enables users to adapt the land cover classification to their specific case study in under 5 minutes.
For updates and access to the code, check my LinkedIn profile regularly. Links to the repository and the Streamlit interface will be provided there.

![image](https://github.com/user-attachments/assets/b5fc3145-7871-4fab-97e4-a22cca15a400)
