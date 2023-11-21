# face_censor_and_display.py
import cv2
import matplotlib.pyplot as plt

def censor_faces_and_display(image_path):
    # 画像読み込み
    image = cv2.imread(image_path)

    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 顔検出用の分類器を読み込み
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 顔の検出
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

    # 検出された顔に対して黒い四角で隠す
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)

    # 画像表示（処理前）
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    # 画像表示（処理後）
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Censored Image')

    plt.show()

if __name__ == "__main__":
    input_image_path = './test.jpg'  # 入力画像のパス

    censor_faces_and_display(input_image_path)
