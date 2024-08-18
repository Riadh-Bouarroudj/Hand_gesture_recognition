import numpy as np
import cv2
import sys
import sqlite3
import os

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap

from keras import models
from skimage.color import rgb2hsv, rgb2ycbcr
from skimage.transform import resize


global user; user=""
global mouvement1; mouvement1=" "
global idi; idi=1 
global path; path=""
global conn

current_directory = os.path.dirname(__file__)
conn=sqlite3.connect(os.path.join(current_directory,'files/Database.db')) 
c=conn.cursor()
c.execute("SELECT * FROM IMAGE where ID=?", (idi, ))
rows=c.fetchall()
listeBDD=[]
   
label2int = {
    'dislike' : 0,
    'like' : 1,
    'ok' : 2,
    'point' : 3,
    'slide_1' : 4,
    'slide_2' : 5,
    'save' : 6
}

def Starting0():
    param=0
    Starting(param)

def Starting1():
    param=1
    Starting(param)
    
def Starting(par=0) :
    global mouvement1
    global user
    print(user)
    global idi
    if par == 0:
        video=cv2.VideoCapture(0)
    if par == 1 :
        Filename2=QFileDialog.getOpenFileName()
        global chemin
        chemin=Filename2[0]
        video=cv2.VideoCapture(chemin)
        
    if par == 1 and chemin =="" :
        print("Error: Could not open video.")
        exit()
    else:
        chemin=""
        cnn=models.load_model(os.path.join(current_directory, 'files/CNNmodel_v1.h5'))
        lstm=models.load_model(os.path.join(current_directory, 'files/LSTMmodel_v1.h5'))

        modelfeatures= models.Model(
            inputs=cnn.input,
            outputs=cnn.get_layer('dense1').output
        )
        
        liste=[]
        li=[]
        lis=[]
        while True :  
            check,frame=video.read()
            if not check:
                print("End of video")
                break
            frame=Treatment(frame)
            liste.append(frame)
            li.append(frame)
    
            if (len(liste)>6):
                lis.append(frame)
                
            
            if (len(lis)==6):
                liste=lis
                lis=[]
                
                
            if (len(liste)==6):
                listeR=[]
                listeR=np.array(liste).reshape(-1,128,128,1)            
        
                listeRR=[]
                listeRR=listeR/255
        
                listeR=[]
                listeR=modelfeatures.predict(listeRR)
                
                listeR=np.array(listeR).reshape(-1,6,150)
                
                y_pred=lstm.predict(listeR)
                
                if ( max(y_pred[0]) >0.85 ) : #Addapt this probability to your need
                    
                    ycla = [np.argmax(element)for element in y_pred ]
                    i=0
                    for i in ycla :
                        for j in label2int:
                            if label2int[j] == i:
                                mouvement = j
                                
                    if mouvement != mouvement1 :
                        mouvement1 = mouvement
                        if mouvement1 =="dislike":
                            ui.comboBox.setCurrentText("Dislike")
                        if mouvement1 =="like":
                            ui.comboBox.setCurrentText("Like")
    
                        if mouvement1 =="ok":
                            ui.comboBox.setCurrentText("Ok")
                            
                        if mouvement1 =="point":
                            ui.comboBox.setCurrentText("Point")
                            
                        if mouvement1 =="slide_1":
                            ui.comboBox.setCurrentText("Slide right")
                            
                        if mouvement1 =="slide_2":
                            ui.comboBox.setCurrentText("Slide left")
                            
                        if mouvement1 =="save":
                            ui.comboBox.setCurrentText("Save")
                            
                        if mouvement1 =="point":
                            ui.comboBox.setCurrentText("Point")
                            
                        print(y_pred)
                        print(mouvement1)
                else:
                    if mouvement1!="Nothing":   
                        mouvement1="Nothing"
                        ui.comboBox.setCurrentIndex(0)
                        ui.warn.setText("")
                        print(mouvement1)
                    
                                
        
        
            key=cv2.waitKey(1)
            if key == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()
            
def Treatment(image):
    cv2.imshow("Press Q to stop the video",image)
    image=cv2.cvtColor(image , cv2.COLOR_BGR2RGB)  
    
    if image.ndim==3:
                
        image = cv2.GaussianBlur(image, (5,5) , 100)
        red , green , bleu = image[:,:,0] ,image[:,:,1] , image[:,:,2]

        """
        #HSV+RBGA mask for hand extraction
        image_hsv = rgb2hsv(image)
        hue = image_hsv[:,:,0]
        sat = image_hsv[:,:,1]
        mask = (hue>=0) & (hue<=0.13888) & (sat>=0.23) & (sat<=0.68) & (red>95) & (green > 40) & (blue > 20) & (red > green) & (red > blue) & (np.abs(red - green) > 15)
        """
        #YCBCR+RGBA mask for hand extraction
        image_ycbcr = rgb2ycbcr(image)
        y = image_ycbcr[:,:,0]
        cb = image_ycbcr[:,:,1]
        cr = image_ycbcr[:,:,2]

        mask = (red > 95) & (green > 40) & (bleu > 20) & (red > green) & (red > bleu) & (np.abs(red - green) > 15) & (cr > 135) & (cb > 85) & (y > 80) & (cr <= (1.5862*cb)+20) & (cr >= (0.3448*cb)+76.2069) & (cr >= (-4.5652*cb)+234.5652) & (cr <= (-1.15 * cb)+301.75) & (cr <= (-2.2857*cb)+432.85)
        mask = mask.astype(np.uint8)
        mask[mask == 1] = 255
        np.sum(mask == 255)

        kernel = np.ones((5,5), np.uint8)
        mask1 = cv2.erode(mask , kernel , iterations=2)

        kernel = np.ones((5,5), np.uint8)
        image = cv2.dilate(mask1 , kernel , iterations=2)
        #cv2.imshow("capturing",image)

        image=resize(image,(128,128))
        return(image)
    else :
        return('The image dimentions do not match')  


def Color_button(user,idi,page):
    if page==1:
        bouton1=ui.button
        botton2=ui.button_2
        botton3=ui.button_3
        botton4=ui.button_4
        botton5=ui.button_5
    elif page==2:
        bouton1=ui.pushButton
        botton2=ui.pushButton_2
        botton3=ui.pushButton_3
        botton4=ui.pushButton_4
        botton5=ui.pushButton_5
    
    if page==1 or page==2:
        d=conn.cursor()
        d.execute("SELECT * FROM Like WHERE Username=? AND IDpic=?",(user,idi))
        rowss=d.fetchall()
        d.close()
        if rowss!=[]:
            bouton1.setStyleSheet("background-color : skyblue")
        else:
            bouton1.setStyleSheet("background-color : None")
            
        d=conn.cursor()
        d.execute("SELECT * FROM Dislike WHERE Username=? AND IDpic=?",(user,idi))
        rowss=d.fetchall()
        d.close()
        if rowss!=[]:
            botton2.setStyleSheet("background-color : red")
        else:
            botton2.setStyleSheet("background-color : None")
            
        d=conn.cursor()
        d.execute("SELECT * FROM Ok WHERE Username=? AND IDpic=?",(user,idi))
        rowss=d.fetchall()
        d.close()
        if rowss!=[]:
            botton3.setStyleSheet("background-color : green")
        else:
            botton3.setStyleSheet("background-color : None")
             
        d=conn.cursor()
        d.execute("SELECT * FROM Save WHERE Username=? AND IDpic=?",(user,idi))
        rowss=d.fetchall()
        d.close()
        if rowss!=[]:
            botton4.setStyleSheet("background-color : grey")
        else:
            botton4.setStyleSheet("background-color : None")
            
        d=conn.cursor()
        d.execute("SELECT * FROM Point WHERE Username=? AND IDpic=?",(user,idi))
        rowss=d.fetchall()
        d.close()
        if rowss!=[]:
            botton5.setStyleSheet("background-color : yellow")
        else:
            botton5.setStyleSheet("background-color : None")
            
def slide_right() :
    ui.warn.setText(" ")
    global idi
    if idi != 7:        
        idi=idi+1
    else:
        idi=1
    ui.warn.setText("Action: Slide right")
    Action(idi,"")
    
def slide_left() :
    ui.warn.setText(" ")
    global idi
    if idi != 1:        
        idi=idi-1
    else:
        idi=7
    ui.warn.setText("Action: Slide left")
    Action(idi,"")
  
  
def Action(idi,classe="",page=0 ):
    Classe_to_Id= {
    'Like' : 2,
    'Dislike' : 3,
    'Ok' : 4,
    'Save' : 5,
    'Point' : 6
}
    global user
    global conn
    c=conn.cursor()
    c.execute("SELECT * FROM IMAGE where ID=?", (idi, ))
    rows=c.fetchall()
    listeBDD=[]
    for row in rows:
        for i in range(0,7):
            listeBDD.append(row[i])
            
    with open (os.path.join(current_directory, 'files/picture.jpg'),'wb') as f :
        f.write(listeBDD[1])   
    conn.commit()
    c.close()
             
    if user =="":
        if classe!= "" and classe!= "Nothing":
            ui.warn.setText("Please log in to perform the action "+classe)
    else: 
        if classe== "Like" or classe== "Dislike" or classe== "Ok":
            d=conn.cursor()
            d.execute("SELECT * FROM "+classe+" WHERE Username=? AND IDpic=?",(user,idi))
            rowss=d.fetchall()
            d.close()
            ui.warn.setText("Action:"+classe)
            if rowss!=[]:
                if classe=="Ok":
                    ui.warn.setText("You have already "+classe+"ed this image")
                else:
                    ui.warn.setText("You have already "+classe+"d this image")
            else:
                liste=["Like","Dislike","Ok"]
                find=False
                for classa in liste:
                    if classa!=classe:
                        d=conn.cursor()
                        d.execute("SELECT * FROM "+classa+" WHERE Username=? AND IDpic=?",(user,idi))
                        rowss=d.fetchall()
                        d.close()
                        if rowss!=[]:
                            ui.warn.setText("Reaction changed from "+classa+" to "+classe)
                            listeBDD[Classe_to_Id[classe]]=listeBDD[Classe_to_Id[classe]]+1
                            listeBDD[Classe_to_Id[classa]]=listeBDD[Classe_to_Id[classa]]-1
                            c=conn.cursor()
                            c.execute("UPDATE IMAGE SET "+classe+"=? WHERE ID=?", (listeBDD[Classe_to_Id[classe]],idi ))
                            c.execute("UPDATE IMAGE SET "+classa+"=? WHERE ID=?", (listeBDD[Classe_to_Id[classa]],idi ))
                
                            d=conn.cursor()
                            d.execute("DELETE FROM "+classa+" WHERE Username=? AND IDpic=?",(user,idi))
                            conn.commit()
                            d.close()
                            conn.commit()
                            c.close()
                            
                            r=conn.cursor()
                            r.execute("INSERT INTO "+classe+" VALUES (?,?)", (user,idi))
                            conn.commit()
                            r.close()
                            ["Like","Dislike","Ok"]
                            if classe=="Like":
                                ui.button.setStyleSheet("background-color : skyblue")
                                ui.button_2.setStyleSheet("background-color : None")
                                ui.button_3.setStyleSheet("background-color : None")
                            elif classe=="Dislike":
                                ui.button_2.setStyleSheet("background-color : red")
                                ui.button.setStyleSheet("background-color : None")
                                ui.button_3.setStyleSheet("background-color : None")
                            elif classe=="Ok":
                                ui.button_3.setStyleSheet("background-color : green")
                                ui.button_2.setStyleSheet("background-color : None")
                                ui.button.setStyleSheet("background-color : None")
                            find=True
                            break
                if find==False:
                    listeBDD[Classe_to_Id[classe]]=listeBDD[Classe_to_Id[classe]]+1
                    c=conn.cursor()
                    c.execute("UPDATE IMAGE SET "+classe+"=? WHERE ID=?", (listeBDD[Classe_to_Id[classe]],idi ))
                    conn.commit()
                    c.close()
                    ui.warn.setText("Action:"+classe)
                    
                    r=conn.cursor()
                    r.execute("INSERT INTO "+classe+" VALUES (?,?)", (user,idi))
                    conn.commit()
                    r.close()  

        elif classe== "Save" or classe== "Point":
            d=conn.cursor()
            d.execute("SELECT * FROM "+classe+" WHERE Username=? AND IDpic=?",(user,idi))
            rowss=d.fetchall()
            d.close()
            ui.warn.setText("Action:"+classe)
            
            if rowss!=[]:
                if classe=="Save":
                    ui.warn.setText("You have already "+classe+"d this image")
                else:
                    ui.warn.setText("You have already "+classe+"ed this image")
            else:               
                listeBDD[Classe_to_Id[classe]]=listeBDD[Classe_to_Id[classe]]+1
                c=conn.cursor()
                c.execute("UPDATE IMAGE SET "+classe+"=? WHERE ID=?", (listeBDD[Classe_to_Id[classe]],idi ))
                conn.commit()
                c.close()
                ui.warn.setText("Action:"+classe)
                
                r=conn.cursor()
                r.execute("INSERT INTO "+classe+" VALUES (?,?)", (user,idi))
                conn.commit()
                r.close()

                if classe=="Save":
                    ui.button_4.setStyleSheet("background-color : grey")
                elif classe== "Point":
                    ui.button_5.setStyleSheet("background-color : yellow")            

    if classe != "" :  
        if classe =="Nothing":
            ui.comboBox.setCurrentIndex(0)
            ui.warn.setText(" ")
        else :    
            ui.comboBox.setCurrentText(classe)

    if page==0:
        ui.button.setText("Like : "+str(listeBDD[2]))
        ui.button_2.setText("Dislike : "+str(listeBDD[3]))
        ui.button_3.setText("OK : "+str(listeBDD[4]))
        ui.button_4.setText("Saves : "+str(listeBDD[5]))
        ui.button_5.setText("Points : "+str(listeBDD[6]))
        
        pixmap = QPixmap(os.path.join(current_directory, 'files/picture.jpg'))
        ui.photo.setPixmap(pixmap)
        Color_button(user,idi,1)
    else:
        ui.pushButton.setText("Like : "+str(listeBDD[2]))
        ui.pushButton_2.setText("Dislike : "+str(listeBDD[3]))
        ui.pushButton_3.setText("OK : "+str(listeBDD[4]))
        ui.pushButton_4.setText("Saves : "+str(listeBDD[5]))
        ui.pushButton_5.setText("Points : "+str(listeBDD[6]))
        
        pixmap = QPixmap(os.path.join(current_directory, 'files/picture.jpg'))
        ui.photo_2.setPixmap(pixmap)  
        Color_button(user,idi,2)
 
def Action_loading() :
    if  ui.comboBox.currentText()== "Like" :
        Action(idi,"Like")
        
    if ui.comboBox.currentText() == "Dislike":
        Action(idi,"Dislike")
     
    if ui.comboBox.currentText() == "Ok":
        Action(idi,"Ok")
     
    if ui.comboBox.currentText() == "Save" :
        Action(idi,"Save")
        
    if ui.comboBox.currentText() == "Point":
        Action(idi,"Point")

    if ui.comboBox.currentText() == " " :
        Action(idi,"Nothing")
        
    if ui.comboBox.currentText() == "Slide right":
        slide_right()
              
    if ui.comboBox.currentText() == "Slide left" :
        slide_left()

    
def Sign_in():
    global user
    global conn 
    c=conn.cursor()
    us=ui.Useredit.text()
    c.execute("SELECT * FROM USERS where USERNAME = ?",(us,))
    rows=c.fetchall()
    if rows==[] : 
        msg=QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText("Username not found")
        x=msg.exec_()
    else:  
        for row in rows:
            a=row[1]
        if ui.MDPedit.text()==a:
            Profil_init()
            user=us
            ui.tabWidget.setCurrentIndex(1)
            
            ui.warn.setText("")
            m="Welcome "+user
            ui.message.setText(m)
            q=str(row[2])
            e=str(row[3])
            if q!="":
                ui.message1.setText("First name : "+q)
            if e!="":
                ui.message2.setText("Last name : "+e)
            
            Color_button(user,idi,1)
            ui.Useredit.setText("")
            ui.MDPedit.setText("")
            
        else:
            if ui.MDPedit.text()=="":                
                msg=QMessageBox()
                msg.setWindowTitle("Error")
                msg.setText("Enter your password")
                x=msg.exec_()
            else :
                msg=QMessageBox()
                msg.setWindowTitle("Error")
                msg.setText("Password incorrect")
                x=msg.exec_()                
    c.close()

def Sign_up():
    global user
    global conn
    c=conn.cursor()
    us=ui.Useredit_2.text()
    c.execute("SELECT * FROM USERS where USERNAME = ?",(us,))
    rows=c.fetchall()
    if rows!=[] or us=="" : 
        msg=QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText("Username already used")
        x=msg.exec_()
    else:
        a=ui.MDPedit_2.text()
        if a=="":
            msg=QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Enter a password")
            x=msg.exec_()
            
        else:
            Profil_init()
            b=ui.nomedit.text()
            d=ui.prenomedit.text()
            
            z=conn.cursor()
            z.execute("""INSERT INTO USERS(Username,MDP,Nom,Prenom) VALUES(?,?,?,?)""",(us,a,b,d))        
            conn.commit()
            z.close()
            user=us
            ui.tabWidget.setCurrentIndex(1)
            
            m="Welcome "+user
            ui.message.setText(m)
            ui.message1.setText("First name : "+b)
            ui.message2.setText("Last name : "+d)
            ui.Useredit_2.setText("")
            ui.MDPedit_2.setText("")
            ui.nomedit.setText("")
            ui.prenomedit.setText("")
    c.close()


def Profil_init():
    global user
    user=""
    ui.message.setText(" You are not connected")
    ui.message1.setText("")
    ui.message2.setText("") 
    ui.pushButton.setText("")
    ui.pushButton_2.setText("")
    ui.pushButton_3.setText("")
    ui.pushButton_4.setText("")
    ui.pushButton_5.setText("")
    ui.pushButton.setStyleSheet("background-color : None")
    ui.pushButton_2.setStyleSheet("background-color : None")
    ui.pushButton_3.setStyleSheet("background-color : None")
    ui.pushButton_4.setStyleSheet("background-color : None")
    ui.pushButton_5.setStyleSheet("background-color : None")
    
    pixmap = QPixmap(os.path.join(current_directory, 'files/Black_picture.jpg'))
    ui.photo_2.setPixmap(pixmap)     
    ui.tabWidget.setCurrentIndex(2)     
  
def User_Like():
    global user
    global conn
    global liste
    global xx
    xx=0
    
    c=conn.cursor()
    c.execute("SELECT * FROM LIKE where Username=?", (user, ))
    rows=c.fetchall()
    liste=[]
    listeBDD=[]
    x=len(rows)
    if rows==[]:
        msg=QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText("No liked images")
        x=msg.exec_()
    else:
        for row in rows:
            liste.append(row[1])
        c.close()
        d=conn.cursor()
        d.execute("SELECT * FROM IMAGE where ID=?", (liste[0], ))
        rows=d.fetchall()
        listeBDD=[]
        for row in rows:
            for i in range(0,7):
                listeBDD.append(row[i])
                
        with open (os.path.join(current_directory, 'files/picture.jpg'),'wb') as f :
            f.write(listeBDD[1])   
        d.close()
        
        ui.pushButton.setText("Like : "+str(listeBDD[2]))
        ui.pushButton_2.setText("Dislike : "+str(listeBDD[3]))
        ui.pushButton_3.setText("OK : "+str(listeBDD[4]))
        ui.pushButton_4.setText("Saves : "+str(listeBDD[5]))
        ui.pushButton_5.setText("Points : "+str(listeBDD[6]))
        
        pixmap = QPixmap(os.path.join(current_directory, 'files/picture.jpg'))
        ui.photo_2.setPixmap(pixmap)
        Color_button(user,liste[xx],2)

def User_Dislike():
    
    global user
    global conn
    global liste
    global xx
    xx=0
    c=conn.cursor()
    c.execute("SELECT * FROM DISLIKE where Username=?", (user, ))
    rows=c.fetchall()
    liste=[]
    listeBDD=[]
    x=len(rows)
    if rows==[]:
        msg=QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText("No disliked images")
        x=msg.exec_()
    else:
        for row in rows:
            liste.append(row[1])
        c.close()
        d=conn.cursor()
        d.execute("SELECT * FROM IMAGE where ID=?", (liste[0], ))
        rows=d.fetchall()
        listeBDD=[]
        for row in rows:
            for i in range(0,7):
                listeBDD.append(row[i])
                
        with open (os.path.join(current_directory, 'files/picture.jpg'),'wb') as f :
            f.write(listeBDD[1])   
        d.close()
        
        ui.pushButton.setText("Like : "+str(listeBDD[2]))
        ui.pushButton_2.setText("Dislike : "+str(listeBDD[3]))
        ui.pushButton_3.setText("OK : "+str(listeBDD[4]))
        ui.pushButton_4.setText("Saves : "+str(listeBDD[5]))
        ui.pushButton_5.setText("Points : "+str(listeBDD[6]))
        
        pixmap = QPixmap(os.path.join(current_directory, 'files/picture.jpg'))
        ui.photo_2.setPixmap(pixmap)
        Color_button(user,liste[xx],2)

def User_Ok():
    global user
    global conn
    global liste
    global xx
    xx=0

    c=conn.cursor()
    c.execute("SELECT * FROM OK where Username=?", (user, ))
    rows=c.fetchall()
    liste=[]
    listeBDD=[]
    x=len(rows)
    if rows==[]:
        msg=QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText("No OK images")
        x=msg.exec_()
    else:
        for row in rows:
            liste.append(row[1])
        c.close()
        d=conn.cursor()
        d.execute("SELECT * FROM IMAGE where ID=?", (liste[0], ))
        rows=d.fetchall()
        listeBDD=[]
        for row in rows:
            for i in range(0,7):
                listeBDD.append(row[i])
                
        with open (os.path.join(current_directory, 'files/picture.jpg'),'wb') as f :
            f.write(listeBDD[1])   
        d.close()
        
        ui.pushButton.setText("Like : "+str(listeBDD[2]))
        ui.pushButton_2.setText("Dislike : "+str(listeBDD[3]))
        ui.pushButton_3.setText("OK : "+str(listeBDD[4]))
        ui.pushButton_4.setText("Saves : "+str(listeBDD[5]))
        ui.pushButton_5.setText("Points : "+str(listeBDD[6]))
        
        pixmap = QPixmap(os.path.join(current_directory, 'files/picture.jpg'))
        ui.photo_2.setPixmap(pixmap)  
        Color_button(user,liste[xx],2)

def User_Save():
    global user
    global conn
    global liste
    global xx
    xx=0
    
    c=conn.cursor()
    c.execute("SELECT * FROM Save where Username=?", (user, ))
    rows=c.fetchall()
    liste=[]
    listeBDD=[]
    x=len(rows)
    if rows==[]:
        msg=QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText("No saved images")
        x=msg.exec_()
    else:
        for row in rows:
            liste.append(row[1])
        c.close()
        d=conn.cursor()
        d.execute("SELECT * FROM IMAGE where ID=?", (liste[0], ))
        rows=d.fetchall()
        listeBDD=[]
        for row in rows:
            for i in range(0,7):
                listeBDD.append(row[i])
                
        with open (os.path.join(current_directory, 'files/picture.jpg'),'wb') as f :
            f.write(listeBDD[1])   
        d.close()
        
        ui.pushButton.setText("Like : "+str(listeBDD[2]))
        ui.pushButton_2.setText("Dislike : "+str(listeBDD[3]))
        ui.pushButton_3.setText("OK : "+str(listeBDD[4]))
        ui.pushButton_4.setText("Saves : "+str(listeBDD[5]))
        ui.pushButton_5.setText("Points : "+str(listeBDD[6]))
        
        pixmap = QPixmap(os.path.join(current_directory, 'files/picture.jpg'))
        ui.photo_2.setPixmap(pixmap) 
        Color_button(user,liste[xx],2)
        
def User_Point():
    global user
    global conn
    global liste
    global xx
    xx=0
    
    c=conn.cursor()
    c.execute("SELECT * FROM POINT where Username=?", (user, ))
    rows=c.fetchall()
    liste=[]
    listeBDD=[]
    x=len(rows)
    if rows==[]:
        msg=QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText("No pointed images")
        x=msg.exec_()
    else:
        for row in rows:
            liste.append(row[1])
        c.close()
        d=conn.cursor()
        d.execute("SELECT * FROM IMAGE where ID=?", (liste[0], ))
        rows=d.fetchall()
        listeBDD=[]
        for row in rows:
            for i in range(0,7):
                listeBDD.append(row[i])
                
        with open (os.path.join(current_directory, 'files/picture.jpg'),'wb') as f :
            f.write(listeBDD[1])   
        d.close()
        
        ui.pushButton.setText("Like : "+str(listeBDD[2]))
        ui.pushButton_2.setText("Dislike : "+str(listeBDD[3]))
        ui.pushButton_3.setText("OK : "+str(listeBDD[4]))
        ui.pushButton_4.setText("Saves : "+str(listeBDD[5]))
        ui.pushButton_5.setText("Points : "+str(listeBDD[6]))
        
        pixmap = QPixmap(os.path.join(current_directory, 'files/picture.jpg'))
        ui.photo_2.setPixmap(pixmap) 
        Color_button(user,liste[xx],2)        
              
def User_left():
    if user=="":
        msg=QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText("Please log in to perform the action :Slide left")
        x=msg.exec_()
    else :
        global liste
        global xx
        if xx==0:       
            xx=len(liste)-1
        else :
            xx=xx-1
        Action(liste[xx],"",1)

def User_right():
    if user=="":
        msg=QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText("Please log in to perform the action :Slide right")
        x=msg.exec_()

    else:
        global liste
        global xx
        o=len(liste)-1
        if xx==o:
            xx=0
        else :
            xx=xx+1
        Action(liste[xx],"",1)        


    
app = QApplication(sys.argv)
ui = uic.loadUi(os.path.join(current_directory, 'files/Interface.ui'))
ui.tabWidget.setCurrentIndex(0)

for row in rows:
    for i in range(0,7):
        listeBDD.append(row[i])        
with open (os.path.join(current_directory, 'files/picture.jpg'),'wb') as f :
    f.write(listeBDD[1])
ui.button.setText("Like : "+str(listeBDD[2]))
ui.button_2.setText("Dislike : "+str(listeBDD[3]))
ui.button_3.setText("OK : "+str(listeBDD[4]))
ui.button_4.setText("Saves : "+str(listeBDD[5]))
ui.button_5.setText("Points : "+str(listeBDD[6]))

pixmap = QPixmap(os.path.join(current_directory, 'files/picture.jpg'))
ui.photo.setPixmap(pixmap)   
    
if user=="":
    ui.message.setText(" Your are not connected")

################################################################# 
#Design
ui.label_8.setStyleSheet("font: 30pt") #Log in
ui.label.setStyleSheet("font: 14pt")   #Username   
ui.label_2.setStyleSheet("font: 14pt") #Password

ui.label_7.setStyleSheet("font: 30pt") #Sign in
ui.label_3.setStyleSheet("font: 14pt") #Username
ui.label_4.setStyleSheet("font: 14pt") #Password
ui.label_5.setStyleSheet("font: 14pt") #Firs name
ui.label_6.setStyleSheet("font: 14pt") #Last name
                                    
ui.message.setStyleSheet("font: 20pt") #Profil
ui.message1.setStyleSheet("font: 12pt") #First name
ui.message2.setStyleSheet("font: 12pt") #Last name

ui.warn.setStyleSheet("font: 20pt;color: black;qproperty-alignment: AlignCenter;") 


###################################################"###################   
ui.webcam.clicked.connect(Starting0)
ui.vid.clicked.connect(Starting1)

ui.slide_left.clicked.connect(slide_left)
ui.slide_right.clicked.connect(slide_right)
ui.comboBox.currentTextChanged.connect(Action_loading)
ui.connectbutton.clicked.connect(Sign_in)
ui.signbutton.clicked.connect(Sign_up)
ui.dec.clicked.connect(Profil_init)
ui.like.clicked.connect(User_Like)
ui.dislike.clicked.connect(User_Dislike)
ui.ok.clicked.connect(User_Ok)
ui.save.clicked.connect(User_Save)
ui.point.clicked.connect(User_Point)
ui.right.clicked.connect(User_right)
ui.left.clicked.connect(User_left)
    
ui.show()
app.exec_() 
app.quit()
