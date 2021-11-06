import os
import face_recognition
import cv2
import numpy as np
duong_dan = "data/haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(duong_dan)
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
def Them():
    Ten = input("Nhap ten cua ban: ") # Nhập tên muốn thêm vào dữ liệu
    try:
        os.mkdir("HinhAnh")# Tạo thư mục Hình Ảnh để thêm dữ liệu nếu thư mục chưa có thì sẽ tạo nếu không thì sẽ tạo thư mục còn không sẽ đẩy vào hàm xử lý lỗi (except)
    except:
        pass
    try:
        os.mkdir("HinhAnh/" + Ten)# Tạo thư mục Tên bên trong thư mục Hình Ảnh để thêm dữ liệu nếu thư mục chưa có thì sẽ tạo nếu không thì sẽ tạo thư mục còn không sẽ đẩy vào hàm xử lý lỗi (except)
        cap = cv2.VideoCapture(0)# Mở camera
        for i in range(10):# Hàm sẽ chạy từ 0 đến 9 mục đích để lưu 10 hình ảnh vào thư mục ứng với tên đã tạo
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
def HuanLuyen():
    for ThuMuc in os.listdir("HinhAnh"):
        if ThuMuc + ".npy" not in os.listdir("HinhAnh/" + ThuMuc):
            List = []
            phan_tram = 100 / len(os.listdir("HinhAnh/" + ThuMuc))
            for i, Ten in enumerate(os.listdir("HinhAnh/" + ThuMuc)):
                Dem = 0
                anh = cv2.imread("HinhAnh/" + ThuMuc + "/" + Ten)
                chuyen_doi_mau_anh = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)
                ma_hoa_anh = face_recognition.face_encodings(chuyen_doi_mau_anh)[0]
                for Train in os.listdir("HinhAnh/" + ThuMuc):
                    anh_train = cv2.imread("HinhAnh/" + ThuMuc + "/" + Train)
                    chuyen_doi_mau_anh = cv2.cvtColor(anh_train, cv2.COLOR_BGR2RGB)
                    ma_hoa_anh_train = face_recognition.face_encodings(chuyen_doi_mau_anh)[0]
                    kiem_tra = face_recognition.compare_faces([ma_hoa_anh_train], ma_hoa_anh, tolerance=0.4)
                    if kiem_tra[0] == True:
                        Dem += 1
                List.append(Dem)
                print("Dang ma hoa: " + str(format(phan_tram * (i + 1), ".2f")) + "%")
            index = List.index(max(List))
            for id, Ten in enumerate(os.listdir("HinhAnh/" + ThuMuc)):
                if id != index:
                    os.remove("HinhAnh/" + ThuMuc + "/" + Ten)
                else:
                    anh = cv2.imread("HinhAnh/" + ThuMuc + "/" + Ten)
                    chuyen_doi_mau_anh = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)
                    ma_hoa_anh = face_recognition.face_encodings(chuyen_doi_mau_anh)[0]
                    np.save("HinhAnh/" + ThuMuc + "/" + ThuMuc, ma_hoa_anh)
                    print("Ma hoa thanh cong!")
def NhanDien():
    cap = cv2.VideoCapture(0)
    while True:
        ret, anh = cap.read()
        chuyen_doi_mau_anh = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)
        try:
            nhan_dien_khuon_mat = face_cascade.detectMultiScale(chuyen_doi_mau_anh, scaleFactor=1.1, minNeighbors=5)
            Ten = []
            for i in range(len(nhan_dien_khuon_mat)):
                x, y, w, h = nhan_dien_khuon_mat[i]
                anh_webcam = chuyen_doi_mau_anh[y:y + h, x:x + w]
                ma_hoa_anh_webcam = face_recognition.face_encodings(anh_webcam)[0]
                for ThuMuc in os.listdir("HinhAnh"):
                    ma_hoa_anh_thu_muc = np.load("HinhAnh/" + ThuMuc + "/" + ThuMuc + ".npy")
                    kiem_tra = face_recognition.compare_faces([ma_hoa_anh_webcam], ma_hoa_anh_thu_muc, tolerance=0.4)
                    if kiem_tra[0] == True:
                        Ten.append(ThuMuc)
                        break
                if kiem_tra[0] != True:
                    Ten.append("Khong co ten")
                cv2.rectangle(anh, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(anh, Ten[i], (h, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        except:
            pass
        cv2.imshow("Nhan dien", anh)
        key = cv2.waitKey(1)
        if key == 27:
            break
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
