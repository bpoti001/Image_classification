# Image_Classification. 
#### * Training Data:

Used bing bulk downloader from https://github.com/ostrolucky/Bulk-Bing-Image-downloader to download images for different rooms in houses. 

#### * Feature Extraction


Used VggNet CNN to extract features from images which converts images into 4096 lenght vector. 

#### * Classification

On top of VGG network build a softmax to classify images into 7 different classes. 

#### * K-means clustring. 

Applied k-means clustring using SPARK on the vectors generated from images. Using clustering information like radius of cluster and density, we were able to get similar image information and duplicate image information. 
