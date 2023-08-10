# ----------------------------------------------------------------------------
# Copyright (C) [2023] Byounggun Park
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------

import cv2

def main():
    # 원하는 비율을 물어보기
    width, height = map(int, input("해상도를 입력하세요 (예: 640, 640 또는 1280, 1440): ").split(','))

    # 웹캠을 연다 (0은 첫 번째 카메라를 나타냅니다. 원하시는 카메라 번호를 입력하세요)
    cap = cv2.VideoCapture(0)

    # 확인: 카메라를 제대로 열었는지 테스트
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        exit()

    # 웹캠 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        # 현재 프레임 캡쳐
        ret, frame = cap.read()

        # 이미지를 화면에 출력
        cv2.imshow("Webcam Frame", frame)

        # 종료 키: 'q'를 누르면 루프에서 빠져나옵니다
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 화면 및 창을 닫고, 웹캠 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
