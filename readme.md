#### Contributors

- Bharath Vemula
- Vishal
- Varsha

# Joint Classification for Political Bias Prediction
The use of joint models in predicting multiple attributes simultaneously has been shown to improve performance and reduce training time when correctly modeled. For example, predicting political bias and other attributes such as the topic and source of the news can be done more efficiently and accurately through joint modeling. However, the selection of which attributes to model together, the model architecture, and the choice of text representation are all important factors that can affect the performance of such models. In this work, we study and demonstrate the importance of these factors on a political bias dataset, showing how changing the second attribute being predicted, the model architecture, and the learning representation can impact the performance of bias prediction. This research has important implications for the development of more efficient and accurate models for predicting multiple attributes in news data.

## Steps to run
### To install requirements:
```
conda install --file requirements.txt 
```
### To run train and test:
```
python train.py [model='lstm'] [epochs=50] [hidden_nodes=20] [learning_rate=0.0001] [version='baseline']
```

This project was developerd with Python version 3.7.16
