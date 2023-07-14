#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os.path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array


# In[2]:


import streamlit as st
from PIL import Image
from keras.models import load_model
import requests


# In[3]:


model = load_model('FV.h5')


# In[4]:


labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}


# In[5]:


fruits = ['Apple', 'Banana', 'Bell Pepper', 'Capsicum', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']


# In[6]:


calorie = {'Apple': 48, 'Banana': 89, 'Beetroot': 43, 'Bell Pepper': 28, 'Cabbage': 25, 'Capsicum': 40, 'Carrot': 41,
           'Cauliflower': 25, 'Chilli Pepper': 40, 'Corn': 86, 'Cucumber': 16, 'Eggplant': 25, 'Garlic': 149, 'Ginger': 80, 
           'Grapes': 69, 'Jalapeno': 29, 'Kiwi': 64, 'Lemon': 29, 'Lettuce': 13, 'Mango': 60, 'Onion': 40, 'Orange': 52, 
           'Paprika': 40, 'Pear': 57, 'Peas': 81, 'Pineapple': 50, 'Pomegranate': 83, 'Potato': 73, 'Raddish': 16,
           'Soy Beans': 172, 'Spinach': 23, 'Sweetcorn': 86, 'Sweetpotato': 86, 'Tomato': 18, 'Turnip': 28, 'Watermelon': 30}


# In[7]:


sugar = {'Apple': 10.1, 'Banana': 12.2, 'Beetroot': 6.8, 'Bell Pepper': 2.5, 'Cabbage': 3.2, 'Capsicum': 5.1, 'Carrot': 4.7,
           'Cauliflower': 1.9, 'Chilli Pepper': 5.3, 'Corn': 3.2, 'Cucumber': 1.7, 'Eggplant': 3.5, 'Garlic': 1.0, 'Ginger': 1.7, 
           'Grapes': 15.5, 'Jalapeno': 4.1, 'Kiwi': 9.0, 'Lemon': 2.5, 'Lettuce': 0.94, 'Mango': 13.7, 'Onion': 4.2, 'Orange': 8.6, 
           'Paprika': 5.1, 'Pear': 9.8, 'Peas': 5.7, 'Pineapple': 9.9, 'Pomegranate': 13.7, 'Potato': 0.7, 'Raddish': 1.9,
           'Soy Beans': 3.0, 'Spinach': 0.4, 'Sweetcorn': 3.2, 'Sweetpotato': 4.2, 'Tomato': 2.6, 'Turnip': 3.8, 'Watermelon': 6.2}


# In[8]:


blood_glucose = {'Apple': 'Minimal increase in blood glucose level', 'Banana': 'According to ripeness. Ripe bananas causes a higher increase in blood glucose',
                 'Beetroot': 'Prevent increase in blood glucose level', 'Bell Pepper': 'Does not increase blood glucose level',
                 'Cabbage': 'Helps in maintaining good blood glucose level', 'Capsicum': 'Does not increase blood glucose level',
                 'Carrot': 'Does not increase blood glucose level', 'Cauliflower': 'Does not increase blood glucose level', 'Chilli Pepper': 'Does not increase blood glucose level',
                 'Corn': 'Can increase blood glucose level if taken in huge amount', 'Cucumber': 'Prevent increase in blood glucose level',
                 'Eggplant': 'Does not increase blood glucose level', 'Garlic': 'Does not increase blood glucose level', 'Ginger': 'Helps in maintaining good blood glucose level', 
                 'Grapes': 'Does not increase blood glucose level', 'Jalapeno': 'Does not increase blood glucose level', 'Kiwi': 'Does not increase blood glucose level',
                 'Lemon': 'Can help to reduce blood glucose level', 'Lettuce': 'Can help to reduce blood glucose level',
                 'Mango': 'According to ripeness. Ripe or over-riped mangoes can cause higher increase in blood glucose level',
                 'Onion': 'Can help to reduce blood glucose level', 'Orange': 'Minimal increase in blood glucose level',
                 'Paprika': 'Helps in maintaining good blood glucose level', 'Pear': 'Minimal increase in blood glucose level',
                 'Peas': 'Helps in maintaining good blood glucose level', 'Pineapple': 'Can increase blood glucose level if taken in huge amount',
                 'Pomegranate': 'Minimal increase in blood glucose level', 'Potato': 'Can increase blood glucose level quickly',
                 'Raddish': 'Can help to regulate blood glucose level', 'Soy Beans': 'Can help to regulate blood glucose level',
                 'Spinach': 'Can help to reduce blood glucose level', 'Sweetcorn': 'Can increase blood glucose level if taken in huge amount',
                 'Sweetpotato': 'Can help to regulate blood glucose level', 'Tomato': 'Does not increase blood glucose level',
                 'Turnip': 'Can help to regulate blood glucose level', 'Watermelon': 'Can increase blood glucose level if taken in huge amount'}


# In[9]:


def run(): 
    st.title("🍍🍍Fruits and Vegetables Classification with Calorie and Sugar Content🍅🍅")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = os.path.join('./upload_images', img_file.name)
        img.save(save_image_path)

        # if st.button("Predict"):
        if img_file is not None:
            # result = processed_img(save_image_path)
            url = 'http://192.168.1.5:5000/predict'
            form_data = {'file': open(save_image_path, 'rb')}
            resp = requests.post(url, files=form_data)
            resp_dict = resp.json()
            result = resp_dict['prediction']
            print(result)
            if result in vegetables:
                st.info('**Category : Vegetables**')
            else:
                st.info('**Category : Fruit**')
            st.success("**Name : " + result + '**')
            cal = calorie
            if cal:
                st.warning('**' + cal + '(kcal per 100 grams)**')
            sug = sugar
            if sug:
                st.warning('**' + sug + '(g per 100 grams)')
            blood_glucose


# In[10]:


run()

