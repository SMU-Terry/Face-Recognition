# This project is specifically used for attendance record
# face_recognition.py
Using hog to detect face. Using ResNet to recognize the face. Built a face detector for recording attendance

Using args to select mode, firstly you need to regist the face into the system, so do not forget to type '--mode=reg' after you run the script by python. (PS: you can also type '--name=xx', '--id=xx' to choose the name and id of the face you wanna registered)

![image](https://user-images.githubusercontent.com/64240681/166260518-568916e3-0377-4522-b556-48e0b569b1d5.png)


Regist a face, it will record for three times to improve the accuracy of next detection

![image](https://user-images.githubusercontent.com/64240681/166261358-faf116e3-21a4-4a32-9c90-6e05cc1d3b35.png)

Secendly you can use the face you detect in ref mode to do the recognization part, also do not forget to type '--mode=reg' or do not type any mode(cuase recog mode is default setting) after you run the script by python

![image](https://user-images.githubusercontent.com/64240681/166261617-c06c23ab-ffc7-46d6-b28d-bdc578ed41bf.png)

The proccess of recogniton.....
![image](https://user-images.githubusercontent.com/64240681/166262270-4d063eed-b847-48b8-ac5a-e7ebc8dcd049.png)

This attendance records will be stored in a csv formate file, so that you can check wheather your employer worked at that day
![image](https://user-images.githubusercontent.com/64240681/166262532-1a2db96b-e83b-44ff-a993-fa5d08cc5ed3.png)

# face_detection.py
It's just a script by using hog algorithm to detect your face. But it can not recogize your face




