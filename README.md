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
### 2.3 Tạo hàm Thêm
* Hàm ```Them()``` dùng để lưu lại 10 ảnh khuôn mặt và lưu vào trong cơ sở dữ liệu.
```python
def Them():
    Ten = input("Nhap ten cua ban: ") # Nhập tên muốn thêm vào dữ liệu
    try:
        os.mkdir("HinhAnh")# Tạo thư mục Hình Ảnh để thêm dữ liệu nếu thư mục chưa có thì sẽ tạo nếu không thì sẽ tạo thư mục còn không sẽ đẩy vào hàm xử lý lỗi (except)
    except:
        pass
    try:
        os.mkdir("HinhAnh/" + Ten)# Tạo thư mục Tên bên trong thư mục Hình Ảnh để thêm dữ liệu nếu thư mục chưa có thì sẽ tạo nếu không thì sẽ tạo thư mục còn không sẽ đẩy vào hàm xử lý lỗi (except)
        cap = cv2.VideoCapture(0)# Mở camera
        for i in range(10):# Hàm sẽ chạy từ 0 đến 9 mục đích để lưu 10 ảnh khuôn mặt vào thư mục ứng với tên đã tạo
            ret, anh = cap.read()# Đọc 1 hình ảnh trên camera
            chuyen_doi_anh = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)# Chuyển đổi màu ảnh đã đọc sang dạng RGB (Red-Green-Blue)
            nhan_dien_khuon_mat = face_cascade.detectMultiScale(chuyen_doi_anh, scaleFactor=1.1, minNeighbors=5)# Nhận diện khuôn mặt và trả về các giá trị tạo độ, chiều rộng, chiều dài của khuôn mặt nhận diện được
            for (toa_do_x, toa_do_y, chieu_dai, chieu_rong) in nhan_dien_khuon_mat:# Khởi tạo vòng lặp để lấy các giá trị trả về mục đích để lấy khuôn mặt và ghi vào thư mục
                cat_khuon_mat = anh[toa_do_y:toa_do_y + chieu_dai, toa_do_x:toa_do_x + chieu_rong]# Khi đã có các giá trị và sẽ thực hiện cắt khuôn mặt trong ảnh nhằm mục đích giảm dung lượng lưu trữ
                cv2.imwrite("HinhAnh/" + Ten + "/" + Ten + str(i) + ".JPG", cat_khuon_mat)# Hàm sẽ thực hiện ghi các ảnh vào thư mục
                cv2.rectangle(anh, (toa_do_x, toa_do_y), (toa_do_x + chieu_dai, toa_do_y + chieu_rong), (255, 0, 0),2)# Vẽ hình chữ nhận bao quanh khuôn mặt
                cv2.putText(anh, "Them "+Ten, (chieu_dai // 2, toa_do_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2, cv2.LINE_AA)# Vẽ chữ ứng với tên trên hình chữ nhật
            cv2.imshow("Them anh", anh)# Hiển thị ảnh ra màn hình
    except:
        print("Da co thu muc")
```
* Giải thích:
  * Hàm ```face_cascade.detectMultiScale()``` truyền 2 tham số **scaleFactor** và **minNeighbors** để tìm các khuôn mặt:
    * **scaleFactor** dùng để thay đổi tỉ lệ kích thước của hình ảnh. Tỉ lệ càng giảm thấp thì số lượng tìm được khuôn mặt càng cao. Ví dụ: 1.03 là giảm 3%, 1.05 là giảm 5%. Rủi ro khi cho tỉ lệ thấp sẽ ảnh hưởng đến nhận diện những vị trí không phải khuôn mặt và thời gian để nhận diện sẽ tăng. Thông thường tỉ lệ là 1.1.

    ![ImageScale](https://user-images.githubusercontent.com/88564663/140603742-6cc0731f-5aac-4ebf-9d65-6191ba330029.png)
    
    * **minNeighbors** dùng để tìm các khuôn mặt lân cận theo khoảng cách đã cho. Giá trị càng thấp như 0 hoặc 1 thì tìm được số lượng khuôn mặt càng cao nhưng sẽ nhận được những khuôn mặt bị trùng nhau và thời gian để nhận diện sẽ tăng. Thuông thường giá trị là 5.

**minNeighbors = 0**

![image](https://user-images.githubusercontent.com/88564663/140614875-6c4c6202-fa2d-4cd9-8d89-4d2f6a0d1e11.png)

**minNeighbors = 5**

![image](https://user-images.githubusercontent.com/88564663/140614900-3aa0a9fe-981a-4a4f-8ed4-02932c514e59.png)

### 2.4 Tạo hàm Huấn Luyện
* Hàm ```HuanLuyen()``` dùng để lấy 1 hình ảnh và so sánh với các ảnh cùng thư mục với nó để tìm ra ảnh nào có giá trị đúng lớn nhât ứng với mỗi người có trong cơ sở dữ liệu và mã hoá rồi cho vào file có đuôi .npy. Khi mã hoá sẽ xoá các ảnh có giá trị đúng thấp hơn đi.
```python
def HuanLuyen():
    for ThuMuc in os.listdir("HinhAnh"): # Hàm os.listdir() trả về các tên thư mục con chứa trong thư mục cha
        if ThuMuc + ".npy" not in os.listdir("HinhAnh/" + ThuMuc): # Kiểm tra trong các thư mục con có file đuôi .npy chưa nếu chưa thì bắt đầu huấn luyện
            List = [] # Tạo mảng để lưu các giá trị đúng của mỗi hình ảnh
            phan_tram = 100 / len(os.listdir("HinhAnh/" + ThuMuc)) # Khởi tạo biến phần trăm để xem tiến trình 
            for i, Ten in enumerate(os.listdir("HinhAnh/" + ThuMuc)): # Duyệt các ảnh có trong từng thư mục con
                Dem = 0 # Khởi tạo biến đếm
                anh = cv2.imread("HinhAnh/" + ThuMuc + "/" + Ten) # Đọc hình ảnh
                chuyen_doi_mau_anh = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB) # Chuyển đổi màu ảnh đã đọc sang dạng RGB (Red-Green-Blue)
                ma_hoa_anh = face_recognition.face_encodings(chuyen_doi_mau_anh)[0] # Mã hoá khuôn mặt thành dạng mảng với 128 chiều và lấy khuôn mặt thứ [0] (Chỉ mã hoá được khi ảnh chuyển sang dạng RGB)
                for Train in os.listdir("HinhAnh/" + ThuMuc): # Duyệt các ảnh có trong từng thư mục con
                    anh_train = cv2.imread("HinhAnh/" + ThuMuc + "/" + Train) # Đọc hình ảnh
                    chuyen_doi_mau_anh = cv2.cvtColor(anh_train, cv2.COLOR_BGR2RGB) # Chuyển đổi màu ảnh đã đọc sang dạng RGB (Red-Green-Blue)
                    ma_hoa_anh_train = face_recognition.face_encodings(chuyen_doi_mau_anh)[0] # Mã hoá khuôn mặt thành dạng mảng với 128 chiều và lấy khuôn mặt thứ [0] (Chỉ mã hoá được khi ảnh chuyển sang dạng RGB)
                    kiem_tra = face_recognition.compare_faces([ma_hoa_anh_train], ma_hoa_anh, tolerance=0.4) # So sánh 2 khuôn mặt đã được mã hoá ứng với sai số 0.4 (tolerance)
                    if kiem_tra[0] == True: #Khi 2 khuôn mặt trả về giá trị đúng (True) thì sẽ tăng biến đếm lên 1
                        Dem += 1 
                List.append(Dem) # Khi thực hiện so sánh 1 ảnh với các ảnh xong thì sẽ thêm giá trị đếm vào danh sách
                print("Dang ma hoa: " + str(format(phan_tram * (i + 1), ".2f")) + "%") # Hiển thị tiến trình
            index = List.index(max(List)) # Khi đã hoàn thành tiến trình thì sẽ lấy vị trí có giá trị lớn nhất trong mảng và truyền vào biến index
            for id, Ten in enumerate(os.listdir("HinhAnh/" + ThuMuc)): # Duyệt các ảnh có trong từng thư mục con
                if id != index:# Kiểm tra giá trị vị trí ảnh có trùng với giá trị index hay không nếu không trùng thì sẽ xoá hình ảnh
                    os.remove("HinhAnh/" + ThuMuc + "/" + Ten) 
                else: # Nếu trùng với index sẽ đọc hình ảnh và mã hoá rồi cho vào file .npy
                    anh = cv2.imread("HinhAnh/" + ThuMuc + "/" + Ten) # Đọc hình ảnh
                    chuyen_doi_mau_anh = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB) # Chuyển đổi màu ảnh đã đọc sang dạng RGB (Red-Green-Blue)
                    ma_hoa_anh = face_recognition.face_encodings(chuyen_doi_mau_anh)[0] # Mã hoá khuôn mặt thành dạng mảng với 128 chiều và lấy khuôn mặt thứ [0] (Chỉ mã hoá được khi ảnh chuyển sang dạng RGB)
                    np.save("HinhAnh/" + ThuMuc + "/" + ThuMuc, ma_hoa_anh) # Lưu mảng vào file đuôi .npy và đưa vào thư mục ứng với mỗi người
                    print("Ma hoa thanh cong!") # Hiển thị đã mã hoá thành công
```

### 2.5 Tạo hàm Nhận Diện
* Hàm ```NhanDien()``` sẽ mở Camera và hiển thị tên tương ứng với khuôn mặt được tìm thấy nếu không tìm thấy sẽ hiển thị không có tên.
```python
def NhanDien():
    cap = cv2.VideoCapture(0)
    while True:
        ret, anh = cap.read()
        chuyen_doi_mau_anh = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)
        try:
            nhan_dien_khuon_mat = face_cascade.detectMultiScale(chuyen_doi_mau_anh, scaleFactor=1.1, minNeighbors=5)
            Ten = [] # Tạo mảng tên để lưu các tên cần hiển thị
            for i in range(len(nhan_dien_khuon_mat)):
                x, y, w, h = nhan_dien_khuon_mat[i]
                anh_webcam = chuyen_doi_mau_anh[y:y + h, x:x + w] # Giới hạn khuôn mặt để giúp mã hoá nhanh hơn thay vì cả khung hình
                ma_hoa_anh_webcam = face_recognition.face_encodings(anh_webcam)[0] 
                for ThuMuc in os.listdir("HinhAnh"):
                    ma_hoa_anh_thu_muc = np.load("HinhAnh/" + ThuMuc + "/" + ThuMuc + ".npy") # Nạp các khuôn mặt đã mã hoá
                    kiem_tra = face_recognition.compare_faces([ma_hoa_anh_webcam], ma_hoa_anh_thu_muc, tolerance=0.4)
                    if kiem_tra[0] == True: # Kiểm tra nếu đúng khuôn mặt thì sẽ thêm vào mảng Tên và thoát khỏi vòng lặp
                        Ten.append(ThuMuc)
                        break
                if kiem_tra[0] != True: # Khi thực hiện kiểm tra xong mà kết quả trả về không đúng thì sẽ thêm Không Có Tên vào mảng Tên
                    Ten.append("Khong co ten")
                cv2.rectangle(anh, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(anh, Ten[i], (h, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        except:
            pass
        cv2.imshow("Nhan dien", anh)
        key = cv2.waitKey(1) # Hàm cv2.waitKey() truyền vào tham số là 1 thì sẽ nhận các sự kiện từ bàn phím
        if key == 27: # Ấn ESC (27) thì sẽ thoát
            break
```
## 3. Tổng kết
* Mô hình nhận diện khuôn mặt phần lớn nhận diện với người lớn và nó không hoạt động tốt đối với trẻ con.
* Đối với những bài toán thực tế thì độ chính xác sẽ không thể đúng 100%. Các giá trị của **scaleFactor** và **minNeighbors** sẽ phải thay đổi để có thể nhận diện đúng với từng dự án khác nhau.
## 4. Cảm ơn
* Cảm ơn bạn đã đọc và xem dự án của mình.

