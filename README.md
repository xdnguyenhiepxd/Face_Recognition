# Nhận diện khuôn mặt sử dụng ngôn ngữ Python
## 1. Cài đặt các thư viện
```python
import os # Thư viện OS sẽ dùng để Tạo, Đọc, Xoá thư mục
import face_recognition # Thư viện sẽ dùng để Mã Hoá Khuôn Mặt, So Sánh Khuôn Mặt 
import cv2 # Thư viện sẽ dùng để Đọc, Ghi, Hiển Thị ảnh và Vẽ Khuôn Mặt
import numpy as np # Thư viện sẽ dùng để tạo ra file có đuôi .npy khi đã huấn luyện thành công 1 khuôn mặt
```
* Lưu ý:
  * Để sử dụng thư viện face_recognition đều đầu tiên cần phải cài đặt các thư viện theo thứ tự sau.
    * **cmake**
    * **dlib**: Thư viện **dlib** thường xảy ra lỗi khi cài đặt các phiên bản nên cần cài đặt theo phiên bản Python tương ứng, đối với phiên bản Python 3.6.8 thì sẽ cài đặt phiên bản 19.7.0.
### 1.1 Cài bộ thư viện
**- Để nhận diện khuôn mặt ta cần cài bộ nhận diện đã được đào tạo của thư viện của OpenCV theo các bước sau:**
* Bước 1: Mở cửa sổ CMD (Command Prompt).
* Bước 2: Gõ lệnh ```python``` hoặc ```py``` trên CMD.
* Bước 3: Gõ lệnh ```import cv2``` (Để gọi được thư viện cv2 ta cần cài đặt thư viện **opencv-python**).
* Bước 4: Gõ lệnh ```print(cv2.__file__)``` để hiển thị đường dẫn thư mục cv2. Ví dụ: ```C:\Users\Administrator\AppData\Local\Programs\Python\Python36\lib\site-packages\cv2\__init__.py```.
* Bước 5: Copy đường dẫn (Bỏ đường dẫn cuối ```__init__.py```). Ví dụ: ```C:\Users\Administrator\AppData\Local\Programs\Python\Python36\lib\site-packages\cv2\```.
* Bước 6: Copy thư mục **data** vào thư mục nhận diện khuôn mặt của bạn hoặc có thể bỏ qua bước này.

**- Hướng dẫn**
<img src="https://user-images.githubusercontent.com/88564663/140616349-42e7ace0-818d-46bd-b016-143c26fa1c79.gif" width="1000"/>

**- Thư viện nhận diện khuôn mặt**

![thumucdata](https://user-images.githubusercontent.com/88564663/140616204-dace98b4-df3a-4b46-9dc6-ea171ef3b422.png)

* Theo như lý thuyết thì độ chính xác phát hiện khuôn mặt phụ thuộc rất nhiều vào file cascade và các tham số. Những file cascade được cung cấp sẵn theo lib OpenCV để phát hiện khuôn mặt có sự khác biệt nhau. Mình thử với ảnh 80 ảnh chứa 77 khuôn mặt thì phát hiện được số lượng khuôn mặt của các bộ nhận diện.
* Trong thư mục **data** sẽ có nhiều thư viện nhưng ta quan tâm vài thư viện sau:
  * **haarcascade_frontalface_alt.xml** 86 khuôn mặt/80 ảnh.
  * **haarcascade_frontalface_alt2.xml** 95 khuôn mặt/80 ảnh.
  * **haarcascade_frontalface_alt_tree.xml** 74 khuôn mặt/80 ảnh.
  * **haarcascade_frontalface_default.xml** 155 khuôn mặt/80 ảnh.
* Như vậy, có thể dùng **haarcascade_frontalface_alt.xml** hoặc **haarcascade_frontalface_alt_tree.xml** để đạt kết quả tốt nhất. Tuy nhiên vẫn còn tuỳ thuộc tập dữ liệu và tham số truyền vào. Do đó cần test cẩn thận bộ dữ liệu sẽ sử dụng.

**- Thêm bộ thư viện vào dự án**
* Sử dụng hàm ```cv2.CascadeClassifier()``` để nạp bộ thư viện.
```python
duong_dan = "data/haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(duong_dan)
```

## 2. Cài đặt chương trình
### 2.1 Tạo hàm Main
* Hàm ```Main()``` dùng để gọi các hàm của chương trình.
```python
def Main():
    while True:
        T= TuyChon()
        if T == 1:
            Them()
        if T == 2:
            HuanLuyen()
        if T == 3:
            NhanDien()
        if T == 4:
            break
        cv2.waitKey(0)
        cv2.destroyAllWindows()
Main()
```
### 2.2 Tạo hàm Tuỳ Chọn
* Hàm ```TuyChon()``` sẽ có 4 tuỳ chọn:
  * **Thêm người mới**: Để thêm 1 người vào cơ sở dữ liệu.
  * **Huấn luyện**: Để huấn luyện các khuôn mặt và tìm ra khuôn mặt có tỉ lệ đúng lớn nhất và tạo file ma trận có đuôi .npy ứng với mỗi người trong cơ sở dữ liệu.
  * **Nhận diện**: Để nhận diện khuôn mặt và hiển thị tên ứng với mỗi khuôn mặt tìm thấy.
  * **Thoát**: Để thoát khỏi chương trình.
* Nếu nhập sai sẽ yêu cầu nhập lại.
```python
def TuyChon():
    print("Tuy chon")
    print("1: Them nguoi moi")
    print("2: Huan luyen")
    print("3: Nhan dien")
    print("4: Thoat")
    try:
        TC = int(input("Tuy chon cua ban la: "))
        return TC
    except:
        print("Vui long nhap lai!")
```
 

![ImageScale](https://user-images.githubusercontent.com/88564663/140603742-6cc0731f-5aac-4ebf-9d65-6191ba330029.png)

![image](https://user-images.githubusercontent.com/88564663/140614875-6c4c6202-fa2d-4cd9-8d89-4d2f6a0d1e11.png)

![image](https://user-images.githubusercontent.com/88564663/140614900-3aa0a9fe-981a-4a4f-8ed4-02932c514e59.png)

