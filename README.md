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

![thumucdata](https://user-images.githubusercontent.com/88564663/140616204-dace98b4-df3a-4b46-9dc6-ea171ef3b422.png)




![ImageScale](https://user-images.githubusercontent.com/88564663/140603742-6cc0731f-5aac-4ebf-9d65-6191ba330029.png)

![image](https://user-images.githubusercontent.com/88564663/140614875-6c4c6202-fa2d-4cd9-8d89-4d2f6a0d1e11.png)

![image](https://user-images.githubusercontent.com/88564663/140614900-3aa0a9fe-981a-4a4f-8ed4-02932c514e59.png)

