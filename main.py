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
    Ten = input("Nhap ten cua ban: ")
    try:
        os.mkdir("HinhAnh")
    except:
        pass
    try:
        os.mkdir("HinhAnh/" + Ten)
        cap = cv2.VideoCapture(0)
        for i in range(10):
            ret, anh = cap.read()
            chuyen_doi_anh = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)
            nhan_dien_khuon_mat = face_cascade.detectMultiScale(chuyen_doi_anh, scaleFactor=1.1, minNeighbors=5)
            for (toa_do_x, toa_do_y, chieu_dai, chieu_rong) in nhan_dien_khuon_mat:
                cat_khuon_mat = anh[toa_do_y:toa_do_y + chieu_dai, toa_do_x:toa_do_x + chieu_rong]
                cv2.imwrite("HinhAnh/" + Ten + "/" + Ten + str(i) + ".JPG", cat_khuon_mat)
                cv2.rectangle(anh, (toa_do_x, toa_do_y), (toa_do_x + chieu_dai, toa_do_y + chieu_rong), (255, 0, 0),2)
                cv2.putText(anh, "Them "+Ten, (chieu_dai // 2, toa_do_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2, cv2.LINE_AA)
            cv2.imshow("Them anh", anh)# Hiển thị ảnh ra màn hình
    except:
        print("Da co thu muc")
def HuanLuyen():
    try:
        for ThuMuc in os.listdir("HinhAnh"):  # Hàm os.listdir() trả về các tên thư mục con chứa trong thư mục cha
            if ThuMuc + ".npy" not in os.listdir("HinhAnh/" + ThuMuc):  # Kiểm tra trong các thư mục con có file đuôi .npy chưa nếu chưa thì bắt đầu huấn luyện
                List = []  # Tạo mảng để lưu các giá trị đúng của mỗi hình ảnh
                phan_tram = 100 / len(os.listdir("HinhAnh/" + ThuMuc))  # Khởi tạo biến phần trăm để xem tiến trình
                for i, Ten in enumerate(os.listdir("HinhAnh/" + ThuMuc)):  # Duyệt các ảnh có trong từng thư mục con
                    Dem = 0  # Khởi tạo biến đếm
                    anh = cv2.imread("HinhAnh/" + ThuMuc + "/" + Ten)  # Đọc hình ảnh
                    chuyen_doi_mau_anh = cv2.cvtColor(anh,cv2.COLOR_BGR2RGB)  # Chuyển đổi màu ảnh đã đọc sang dạng RGB (Red-Green-Blue)
                    ma_hoa_anh = face_recognition.face_encodings(chuyen_doi_mau_anh)[0]  # Mã hoá khuôn mặt thành dạng mảng với 128 chiều và lấy khuôn mặt thứ [0] (Chỉ mã hoá được khi ảnh chuyển sang dạng RGB)
                    for Train in os.listdir("HinhAnh/" + ThuMuc):  # Duyệt các ảnh có trong từng thư mục con
                        anh_train = cv2.imread("HinhAnh/" + ThuMuc + "/" + Train)  # Đọc hình ảnh
                        chuyen_doi_mau_anh = cv2.cvtColor(anh_train,cv2.COLOR_BGR2RGB)  # Chuyển đổi màu ảnh đã đọc sang dạng RGB (Red-Green-Blue)
                        ma_hoa_anh_train = face_recognition.face_encodings(chuyen_doi_mau_anh)[0]  # Mã hoá khuôn mặt thành dạng mảng với 128 chiều và lấy khuôn mặt thứ [0] (Chỉ mã hoá được khi ảnh chuyển sang dạng RGB)
                        kiem_tra = face_recognition.compare_faces([ma_hoa_anh_train], ma_hoa_anh,tolerance=0.4)  # So sánh 2 khuôn mặt đã được mã hoá ứng với sai số 0.4 (tolerance)
                        if kiem_tra[0] == True:  # Khi 2 khuôn mặt trả về giá trị đúng (True) thì sẽ tăng biến đếm lên 1
                            Dem += 1
                    List.append(Dem)  # Khi thực hiện so sánh 1 ảnh với các ảnh xong thì sẽ thêm giá trị đếm vào danh sách
                    print("Dang ma hoa: " + str(format(phan_tram * (i + 1), ".2f")) + "%")  # Hiển thị tiến trình
                index = List.index(max(List))  # Khi đã hoàn thành tiến trình thì sẽ lấy vị trí có giá trị lớn nhất trong mảng và truyền vào biến index
                for id, Ten in enumerate(os.listdir("HinhAnh/" + ThuMuc)):  # Duyệt các ảnh có trong từng thư mục con
                    if id != index:  # Kiểm tra giá trị vị trí ảnh có trùng với giá trị index hay không nếu không trùng thì sẽ xoá hình ảnh
                        os.remove("HinhAnh/" + ThuMuc + "/" + Ten)
                    else:  # Nếu trùng với index sẽ đọc hình ảnh và mã hoá rồi cho vào file .npy
                        anh = cv2.imread("HinhAnh/" + ThuMuc + "/" + Ten)  # Đọc hình ảnh
                        chuyen_doi_mau_anh = cv2.cvtColor(anh,cv2.COLOR_BGR2RGB)  # Chuyển đổi màu ảnh đã đọc sang dạng RGB (Red-Green-Blue)
                        ma_hoa_anh = face_recognition.face_encodings(chuyen_doi_mau_anh)[0]  # Mã hoá khuôn mặt thành dạng mảng với 128 chiều và lấy khuôn mặt thứ [0] (Chỉ mã hoá được khi ảnh chuyển sang dạng RGB)
                        np.save("HinhAnh/" + ThuMuc + "/" + ThuMuc,ma_hoa_anh)  # Lưu mảng vào file đuôi .npy và đưa vào thư mục ứng với mỗi người
                        print("Ma hoa thanh cong!")  # Hiển thị đã mã hoá thành công
    except:
        pass
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
