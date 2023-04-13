# import socket
# import os
# import time 

# files1 = []
# files2 = []

# path = "./wav파일들2"

# # 서버 IP와 포트 설정
# HOST = '0.0.0.0'  # 모든 IP에서 접속 허용
# PORT = 12345  # 사용할 포트 번호

# # 소켓 생성 및 바인딩
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind((HOST, PORT))
# s.listen(5)
# print(f'Server is running on {HOST}:{PORT}')

# while True:
#     # 클라이언트로부터 연결 요청 수락
#     conn, addr = s.accept()
#     print(f'Connected by {addr}')
#     files1 = os.listdir(path)
#     for i in files1 :
#         with open('./wav파일들2/{}'.format(i), 'rb') as f:
#             data = f.read(1024)
#             while data:
#                 conn.send(data)
#                 data = f.read(1024)
#         print('File transmission completed')
#         time.sleep(5)
# #     while 1 :
# #         # 파일 전송
# #         files2 = os.listdir(path)
# #         if files2 != files1 :
# #             for i in files2 :
# #                 if i not in files1:
# #                     with open('./wav파일들2/{}'.format(i), 'rb') as f:
# #                         data = f.read(1024)
# #                         while data:
# #                             conn.send(data)
# #                             data = f.read(1024)
# #                     print('File transmission completed')
# #                     time.sleep(5)
# #             files1 = files2
#     conn.close()

############################################3
import socket
import os
# 서버 IP와 포트 설정
HOST = '0.0.0.0'  # 모든 IP에서 접속 허용
PORT = 12345  # 사용할 포트 번호
path = './wav파일들2'
files = os.listdir(path)
num = len(files)
# 소켓 생성 및 바인딩
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(5)
print(f'Server is running on {HOST}:{PORT}')

while True:
    # 클라이언트로부터 연결 요청 수락
    conn, addr = s.accept()
    print(f'Connected by {addr}')

    # 파일 전송
    for i in range(num):
        with open('./wav파일들2/wav ({}).wav'.format(i), 'rb') as f:
            data = f.read(1024)
            while data:
                conn.send(data)
                data = f.read(1024)
        print('File transmission completed')
    conn.close()