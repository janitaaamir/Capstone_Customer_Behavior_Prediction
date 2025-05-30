# Capstone_Customer_Behavior_Prediction

## Abstract:
Small businesses often face challenges when it comes to making effective decisions about their marketing strategies. Off-the-shelf tools are either too expensive or too generic for their niche customer base. At the same time, small businesses may lack the resources or technical expertise to utilize their data to the full potential. In this project, we build a custom predictive model for STEELPORT Knife Co., a small business that specializes in premium kitchen knives, to help them better understand their customer base and tailor their marketing. Our approach focuses on two key sub-problems: (1) identifying which existing customers are likely to return, and (2) predicting what they might purchase next. We work with a small, imbalanced dataset of past orders and enrich it with features such as household income and gender. Using the dataset, we train Random Forest, XGBoost, and Neural Network models and evaluate their performance using AUC, precision, and recall. We find that our models achieve non-trivial performance on both sub-problems. This suggests that even with limited data, a custom machine learning approach can help uncover useful insights for small businesses.

## Code Overview:

├── data/ # Processed datasets  
├── models/ # Saved model files   
└── README.md # Project overview  
