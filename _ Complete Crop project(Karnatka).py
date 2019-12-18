#!/usr/bin/env python
# coding: utf-8

# In[9]:


from tkinter import *
#from selenium import webdriver
#from selenium.webdriver.support.ui import Select
import pandas as pd
import numpy as ny
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tkinter import messagebox as ms
from sklearn.metrics import accuracy_score
import seaborn as sns
#from PIL import ImageTk,Image


name=""
root=Tk()
root.geometry('1000x800')
root.title("Farmer Crop Prediction")
tkvar=StringVar(root)

#farm
#img = ImageTk.PhotoImage(Image.open("ffarm.jpg"))
#panel=Label(root,image=img)
#panel.place(x=0,y=0)

#boy
#root.wm_attributes('-alpha', 0.7)


global gstate
list1=""
list2=""
list3=""
ste1=Label(root,text="Select State:",font=('Bold',15),bg="yellow").place(x=350,y=180)
ste5=Label(root,text="Recommendation of Crop",font=('Bold',20),bg="white").place(x=400,y=100)
list1={'Andra Pradesh','Aasam','Bihar','Gujarat','Hydrabad','Himachal Pradesh','Karnataka','Kerala','Madhya Pradesh','Maharashtra','Orissa','Punjab','Tamil Nadu','Uttar Pradesh','West Bengal','Chattisgarh','Jammu & Kashmir','Uttarakhand','Rajasthan'}

mnth1=Label(root,text="Select Month: ",font=('Bold',15),bg="yellow").place(x=350,y=230)
list2={'January','February','March','April','May','June','July','August','September','October','November','December'}


sl1=Label(root,text="Select Soil Type:",font=('Bold',15),bg="yellow").place(x=350,y=280)
list3={'Alluvial','Red','Black','Mountain','Laterite','Desert'}

df = pd.read_csv(r'C:\Users\User26\Downloads\cropdata1.csv')

class farm:
#for state
    
    def __init__(self):
        self.name=" "
        
        
    
    def statecode(val):
        global statevalue
        statevalue=val
        print("You have Selected State:",statevalue)
        
    c=StringVar()

#droplist1=0
    c.set("State")
    

    droplist1=OptionMenu(root,c,*list1,command=statecode)
    droplist1.config(width="25")



    abc=c.get()
    droplist1.place(x=600,y=180)

        #for month
        #monthvalue=""
    def monthcode(val):
        global monthvalue
        monthvalue=val
        print("You have Selected month:",monthvalue)
    
    c1=StringVar()
        #droplist.place(x=300,y=90)
    droplist2=OptionMenu(root,c1,*list2,command=monthcode)
    droplist2.config(width="25")
    c1.set("Month")
    abc1=c1.get()
    droplist2.place(x=600,y=230)

        #for soil
        #soilvalue=""
    def soilcode(val):
        global soilvalue
        soilvalue=val
        print("You have Selected Soil type:",soilvalue)
    
    c2=StringVar()

    #droplist.place(x=300,y=90)
    droplist3=OptionMenu(root,c2,*list3,command=soilcode)
    droplist3.config(width="25")
    c2.set("Soil type")
    abc3=c2.get()
    droplist3.place(x=600,y=280)

    
        # print(df)




   

    def predict1(self):
        X = df[['State', 'Month', 'Soil']]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = RandomForestClassifier(n_estimators=450)

        clf.fit(X_train, y_train)
        
        #xtest
        ypred=clf.predict(X_test)
        print(accuracy_score(y_test, ypred))
        
        ste = {'Andra Pradesh': 1, 'Aasam': 2, 'Bihar': 3, 'Gujarat': 4, 'Hydrabad': 5, 'Himachal Pradesh': 6,
               'Karnataka': 7, 'Kerala': 8, 'Madhya Pradesh': 9, 'Maharashtra': 10, 'Orissa': 11, 'Punjab': 12,
               'Tamil Nadu': 13, 'Uttar Pradesh': 14, 'West Bengal': 15, 'Chattisgarh': 16, 'Jammu & Kashmir': 17,
               'Uttarakhand': 18, 'Rajasthan': 19}
        mnth = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12}
        sl = {'Alluvial': 1, 'Red': 2, 'Black': 3, 'Mountain': 4, 'Laterite': 5, 'Desert': 6}
        Tar = {1: 'Rice', 2: 'Wheat', 3: 'Cotton', 4: 'Sugarcane', 5: 'Tea', 6: 'Ragi', 7: 'Maize', 8: 'Milet', 9: 'Barley',
               12: 'Rice Wheat', 13: 'Rice Cotton', 14: 'Rice Sugarcane', 15: 'Rice Tea', 16: 'Rice Ragi', 17: 'Rice Maize',
               18: 'Rice Milet', 19: 'Rice Barley', 21: 'Wheat Rice', 23: 'Rice Cotton', 24: 'Wheat Sugarcane',

               25: 'Wheat Tea', 26: 'Wheat Ragi', 27: 'Wheat Maize', 28: 'Wheat Milet', 29: 'Wheat Barley',
               31: 'Cotton Rice', 32: 'Cotton Wheat', 34: 'Cotton Sugarcane', 35: 'Cotton Tea', 36: 'Cotton Ragi',
               37: 'Cotton Maize', 38: 'Cotton Milet', 39: 'Cotton Barley', 41: 'Sugarcane Rice', 42: 'Sugarcane Wheat',
               43: 'Sugarcane Cotton', 44: 'Sugarcane Tea', 46: 'Sugarcane Ragi', 47: 'Sugarcane Maize',
               48: 'Sugarcane Milet', 49: 'Sugarcane Barley', 51: 'Tea Rice', 52: 'Tea Wheat', 53: 'Tea Cotton',
               54: 'Tea Sugarcane', 56: 'Tea Ragi', 57: 'Tea Maize', 58: 'Tea Milet', 59: 'Tea Barley', 61: 'Ragi Rice',
               62: 'Ragi Wheat', 63: 'Ragi Cotton', 64: 'Ragi Sugarcane', 65: 'Ragi Tea', 67: 'Ragi Maize',
               68: 'Ragi Milet', 69: 'Ragi Barley', 71: 'Maize Rice', 72: 'Maize Wheat', 73: 'Maize Cotton',
               74: 'Maize Sugarcane', 75: 'Maize Tea', 76: 'Maize Ragi', 78: 'Maize Milet', 79: 'Maize Barley',
               81: 'Milet Rice', 82: 'Milet Wheat', 83: 'Milet Cotton', 84: 'Milet Sugarcane', 85: 'Milet Tea',
               86: 'Milet Ragi', 87: 'Milet Maize', 89: 'Milet Barley', 91: 'Barley Rice', 92: 'Barley Wheat',
               93: 'Barley Cotton', 94: 'Barley Sugarcane', 95: 'Barley Tea', 96: 'Barley Ragi', 97: 'Barley Maize',
               98: 'Barley Milet', 134: 'Rice Cotton Sugarcane', 148: 'Rice Sugarcane Milet', 126: 'Rice Wheat Ragi',
               347: 'Cotton Sugarcane Maize', 13478: 'Rice Cotton Sugarcane Maize Milet',
               1478: 'Rice Sugarcane Maize Milet', 1348: 'Rice Cotton Sugarcane Milet'}
        # print(Tar)

        
        for s in ste.fromkeys(ste):
            if s==statevalue:
               
                stv1 = ste[statevalue]

        print(statevalue)

        print(stv1)
        for m in mnth.fromkeys(mnth):
            if m==monthvalue :
               
                mnv1 = mnth[monthvalue]

        mnv1
        print(monthvalue)
        print(mnv1)
        for soi in sl.fromkeys(sl):
            if soi==soilvalue:
                slv1 = sl[soilvalue]

        slv1
        print(soilvalue)
        print(slv1)
       


        xnew = [[stv1, mnv1, slv1]]
        #print(xnew)
        y_pred = clf.predict(xnew)
        y_pred = y_pred.astype(int)
        print(y_pred)
        for y_pred in Tar.fromkeys(y_pred):
            if y_pred:
                #global name
                self.name = Tar[y_pred]
                print("Recommended Crop type for you is:",self.name,"Crop")
            else:
                print(" none of the crop can be taken in the data you have entred state,month and soil type")

        #self.errors = abs(y_pred - y_test)
        #print(self.errors)
        #self.mape = 100 * (self.errors / y_test)
        #print(self.mape)
        #self.accuracy = 100 - ny.mean(self.mape)
        #print("accuracy is :",self.accuracy)

    def clicked(self):
        ms.showinfo("Thank-You", "I hope that the information was useful!")

    def page(self):
        self.predict1()
        self.restore()


        #img = ImageTk.PhotoImage(Image.open("fcrops.jpg"))
        #panel = Label(root, image=img)
        #panel.place(x=0, y=0)

        root.geometry("1200x450")  #700x650
        root.wm_title("Result")
        l = Label(root, text="Recommended Crop type for you is : " + self.name  , font=("", 15))
        l.place(x=500, y=200)
        #root.background=img
        b = Button(root, text="Okay", bg='lightblue', fg="black", font=("bold", 12),command=self.clicked)
        b.place(x=700, y=300)
        btn = Button(root1, text="FeedBack", bg='green', command=self.clicked).place(x=380, y=460)


    def restore(self):
        for widgets in root.winfo_children():
            widgets.destroy()


obj=farm()
btn=Button(root,text="RECOMMENDED CROP!",bg="blue",fg="white",width=20,height=2,command=obj.page).place(x=460,y=400)
root.mainloop()


# In[ ]:




