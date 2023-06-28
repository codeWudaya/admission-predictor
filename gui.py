from tkinter import *
import joblib
from sklearn.preprocessing import StandardScaler 
import numpy as np

sc=StandardScaler()


def show_entry():
    
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())
    p7 = float(e6.get())
    input_data = np.array([[p1, p2, p3, p4, p5, p6, p7]])
    input_data_scaled = sc.fit_transform(input_data)

    model = joblib.load('admission_model')
#     result = model.predict()
    result = model.predict(input_data_scaled)
    
    if result == 1:
        Label(master, text="High Chance of getting admission").grid(row=31)
    else:
        Label(master, text="You may get admission").grid(row=31)
    
master =Tk()
master.title("Graduate Admission Analysis and Prediction")
label = Label(master,text = "Graduate Admission Analysis and Prediction",bg = "black",
               fg = "white").grid(row=0,columnspan=2)

Label(master,text = "Enter Your GRE Score").grid(row=1)
Label(master,text = "Enter Your TOEFL Score").grid(row=2)
Label(master,text = "Enter University Rating 1-5",).grid(row=3)
Label(master,text = "Enter SOP 1-5").grid(row=4)
Label(master,text = "Enter LOR 1-5").grid(row=5)
Label(master,text = "Enter Your CPGA 1-10").grid(row=6)
Label(master,text = "Research 1/0").grid(row=7)


e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)

e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)
e6.grid(row=6,column=1)
e7.grid(row=7,column=1)


Button(master,text="Predict",command=show_entry).grid()

mainloop()