import streamlit as st
import os
import subprocess


# -------------------------------------------------------------------------------------------
def main():
    st.title("Dashboard")
    
    
    
    if st.button("test"):
        print("test")
    
    
    if st.button("run"):
        print("aa")
        script_path = os.path.join(os.getcwd(), "app", "pages",  "yolov7-dashboard-main","sum_copy7.py")
        result = subprocess.run(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("외부 파일 실행 성공")
            print("표준 출력:\n", result.stdout)
        else:
            print("외부 파일 실행 실패")
            print("표준 에러 출력:\n", result.stderr)

    
if __name__ == "__main__":
    try:
    
        main()
    except SystemExit:
        pass
