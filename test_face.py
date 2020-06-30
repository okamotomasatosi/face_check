import face_recognition
import cv2
import os

if __name__ == '__main__':
    #WEBカメラからの取り込み
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cv2.imwrite('./face2/masa.png', cap.read()[1])

    #比較元ファイルの読み込み
    src_img = face_recognition.load_image_file("./face/obama.png")
    #比較元、顔の数値化
    src_img_encoding = face_recognition.face_encodings(src_img)[0]
    # print(src_img_encoding)

    #比較対象顔設定（読み込みと数値化）
    img1="./face2/tosi.png"
    img2="./face2/tosi2.png"
    img3="./face2/tosi3.png"
    img4="./face2/masa.png"

    #画像読み込み
    dest_img1 = face_recognition.load_image_file(img1)
    #数値化
    dest_img_encoding1 = face_recognition.face_encodings(dest_img1)[0]

    dest_img_encoding2 = face_recognition.face_encodings(face_recognition.load_image_file(img2))[0]
    dest_img_encoding3 = face_recognition.face_encodings(face_recognition.load_image_file(img3))[0]
    dest_img_encoding4 = face_recognition.face_encodings(face_recognition.load_image_file(img4))[0]

    #顔比較
    results = face_recognition.compare_faces([src_img_encoding], dest_img_encoding1)
    print(img1 , results)
    results = face_recognition.compare_faces([src_img_encoding], dest_img_encoding2)
    print(img2 , results)
    results = face_recognition.compare_faces([src_img_encoding], dest_img_encoding3)
    print(img3 , results)
    results = face_recognition.compare_faces([src_img_encoding], dest_img_encoding4)
    print(img4 , results)
