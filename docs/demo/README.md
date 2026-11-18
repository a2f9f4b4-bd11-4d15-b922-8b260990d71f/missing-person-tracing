# Bằng chứng demo / Demo evidence

Các tệp trong thư mục này được sinh ra trực tiếp từ hệ thống đang chạy, không
chỉnh sửa bằng tay.

| Tệp | Nội dung |
|---|---|
| `01-ingest.txt` | Nhập 2 camera từ một clip mẫu: 7 tracklet mỗi camera |
| `02-search.txt` | Nội dung kho dữ liệu và kết quả tìm kiếm (phạm vi cam1, cam2) |
| `03-search-full-gallery.txt` | Tìm trên toàn bộ 915 lần xuất hiện, 9 camera |
| `top10.png` | Ảnh truy vấn và 10 ảnh thu nhỏ kết quả, đối chiếu được bằng mắt |

## Kết quả

915 lần xuất hiện → 159 vượt ngưỡng τ = 0,50 → 10 hiển thị cho người kiểm duyệt.

**Đúng người đứng hạng 1 và 2** với điểm 0,902, cách biệt rõ so với 0,718 của
kết quả tiếp theo. Tám kết quả còn lại là người khác — clip mẫu chứa nhiều
người đi bộ, và hai kết quả đến từ Market-1501.

Giá trị của hệ thống nằm ở **thứ hạng**: người cần tìm nằm ngay đầu danh sách,
và người kiểm duyệt xem 10 ảnh thay vì 915.

## Lưu ý

- Ảnh thu nhỏ Market-1501 được **ẩn** trong `top10.png` vì bộ dữ liệu này chỉ
  cấp phép cho mục đích nghiên cứu.
- Ảnh từ webcam (người thật) không được đưa vào repo công khai.
- Ảnh truy vấn được tạo từ ảnh thu nhỏ của chính một tracklet, nên đây là phép
  thử thông mạch chứng minh pipeline hoạt động, không phải phép đo tổng quát
  hoá. Số liệu tổng quát hoá nằm trong `docs/evaluation.md`.
