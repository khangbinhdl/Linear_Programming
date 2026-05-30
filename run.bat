@echo off
chcp 65001 >nul
echo ===================================================
echo   ĐANG KHỞI ĐỘNG ĐỒ ÁN QUY HOẠCH TUYẾN TÍNH
echo ===================================================
echo.
echo [1/2] Đang kiểm tra và cài đặt thư viện cần thiết...
pip install -r requirements.txt
echo.
echo [2/2] Đang khởi động Server...
echo Máy chủ sẽ chạy tại địa chỉ: http://127.0.0.1:8000
echo (Bạn có thể mở trình duyệt và truy cập link trên)
echo.
cd backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
pause
