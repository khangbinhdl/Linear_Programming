@echo off
chcp 65001 >nul
echo ===================================================
echo   ĐANG KHỞI ĐỘNG ĐỒ ÁN QUY HOẠCH TUYẾN TÍNH
echo ===================================================
echo.
echo [1/2] Đang kiểm tra và cài đặt thư viện cần thiết...
pip install -r requirements.txt
echo.
echo [2/2] Đang khởi động Server Streamlit...
echo (Trình duyệt sẽ tự động mở lên ngay sau đây)
echo.
streamlit run streamlit_app.py
pause
