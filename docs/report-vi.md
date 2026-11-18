# Truy tìm người mất tích — Báo cáo dự án

**Đề tài #15, Mini Projects** · Tháng 10/2026

Tìm kiếm người mất tích trong dữ liệu camera giám sát chỉ từ một ảnh tham
chiếu, sử dụng tái định danh (re-identification) toàn thân qua nhiều camera.

Mã nguồn: `https://github.com/<account>/missing-person-tracing`

---

## 1. Bài toán

Khi một người mất tích, những giờ đầu là quan trọng nhất, nhưng việc xem lại
camera giám sát hiện vẫn làm thủ công: một người phải tua qua hàng giờ video
của từng camera. Mặt người trong camera giám sát thường nhỏ, mờ hoặc quay đi,
nên chỉ dùng nhận diện khuôn mặt (face recognition) sẽ bỏ sót phần lớn các lần
xuất hiện.

Dự án này đi theo hướng khác: so khớp người dựa trên **ngoại hình toàn thân** —
quần áo, dáng người, màu sắc. Đặc trưng này vẫn hoạt động ở độ phân giải thấp và
khi khuôn mặt không bao giờ xuất hiện. Từ một ảnh, hệ thống trả về danh sách
rút gọn đã xếp hạng, kèm ảnh thu nhỏ và video clip có thể phát, để người kiểm
duyệt chỉ cần xem vài phút thay vì hàng giờ.

Hệ thống không định danh bất kỳ ai. Nó chỉ trả lời một câu hỏi: *người này có
xuất hiện trong dữ liệu camera hay không, nếu có thì vào lúc nào và ở camera
nào?*

## 2. Hệ thống đã xây dựng

Một pipeline cục bộ viết bằng Python 3.12, chạy hoàn toàn trên một máy tính,
không dùng dịch vụ đám mây.

```
camera / tệp video  →  MediaMTX  →  ingest worker (một tiến trình mỗi camera)
                                    ├ lấy mẫu khung hình ở 10 fps
                                    ├ YOLO26 phát hiện người
                                    ├ ByteTrack theo vết từng người
                                    └ OSNet nhúng tracklet  →  Postgres + pgvector
                                                                      ↓
                                            ảnh tham chiếu → cùng đường nhúng
                                                           → truy vấn cosine
                                                           → ngưỡng + kiểm tra thời gian
                                                           → kết quả xếp hạng trên web UI
```

Đối tượng trung tâm là **tracklet**: một lần xuất hiện liên tục của một người
trên một camera, được rút gọn thành vector ngoại hình 512 chiều cùng một ảnh
thu nhỏ và một clip. Việc tìm kiếm khi đó trở thành truy vấn láng giềng gần
nhất (nearest neighbour) trên tập vector đó.

**Các thành phần.** Ingest (tệp ghi sẵn và RTSP trực tiếp), phát hiện và theo
vết, tái định danh, lưu trữ Postgres/pgvector, dịch vụ tìm kiếm áp dụng ngưỡng
tương đồng và bộ lọc tính khả thi vật lý, giao diện web kết xuất phía server
(FastAPI + Jinja2 + HTMX), và tác vụ xoá dữ liệu theo thời hạn.

| Chỉ số | Giá trị |
|---|---|
| Mã ứng dụng | 1.825 dòng |
| Mã kiểm thử | 2.536 dòng (199 test) |
| Script phụ trợ | 698 dòng |
| Bộ kiểm thử | **195 passed, 4 skipped** |

4 test bị bỏ qua là các kiểm tra chỉ kích hoạt khi có camera tạo ra lần xuất
hiện trong 30 phút trước đó.

## 3. Các yêu cầu

| ID | Yêu cầu | Trạng thái |
|---|---|---|
| R1 | Nhập tệp video kèm tên camera và thời điểm bắt đầu | Hoàn thành |
| R2 | Phát hiện, theo vết mọi người; lưu mỗi lần xuất hiện kèm ảnh thu nhỏ | Hoàn thành |
| R3 | Ảnh tham chiếu trả về 10 kết quả khớp nhất, đã xếp hạng | Hoàn thành |
| R4 | Mỗi kết quả hiển thị camera, thời gian, điểm, ảnh thu nhỏ, liên kết clip | Hoàn thành |
| R5 | Lọc kết quả theo khoảng thời gian và camera | Hoàn thành |
| R6 | Loại bỏ các kết quả bất khả thi về thời gian | Hoàn thành |
| R7 | Nhận video trực tiếp từ hai camera đồng thời | Hoàn thành |
| R8 | Lần xuất hiện mới tìm được trong vòng 10 giây | Hoàn thành — trung vị 2,1 s |
| R9 | Khung thời gian (timeline), mỗi camera một hàng | Hoàn thành |
| R10 | Người kiểm duyệt xác nhận hoặc từ chối kết quả | Hoàn thành |
| R11 | Dữ liệu tự xoá sau số ngày cấu hình | Hoàn thành |
| R12 | So khớp khuôn mặt như một tín hiệu bổ sung | **Không làm** — xem §7 |
| R13 | Tìm kiếm bằng mô tả văn bản | Hoàn thành |

R12 và R13 là hai phương án thay thế nhau; R13 được chọn vì nhận diện khuôn mặt
đóng góp rất ít ở khoảng cách camera mà hệ thống hướng tới, trong khi tìm kiếm
bằng văn bản bổ sung một khả năng hệ thống chưa có.

## 4. Đánh giá

Đo lường theo ba mức, mỗi mức trả lời một câu hỏi khác nhau. Tất cả số liệu đều
do các script trong repo sinh ra và có thể tạo lại bằng `scripts/eval_table.py`.

### Mức 1 — chỉ riêng mô hình re-ID, trên Market-1501

| Trọng số | Rank-1 | Rank-5 | mAP | Số truy vấn |
|---|---|---|---|---|
| `osnet_ain_x1_0` (MSMT17) | **70,35%** | 84,41% | 43,71% | 3.362 |
| `osnet_x1_0` (MSMT17) | 66,75% | 81,02% | 40,52% | 3.362 |

**Đây là đánh giá xuyên miền (cross-domain).** Trọng số được huấn luyện trên
MSMT17 và đánh giá trên Market-1501, nên mốc so sánh đúng là các kết quả
xuyên miền trong tài liệu khoa học, **không phải** con số rank-1 khoảng 94%
thường được dẫn — con số đó giả định mô hình được huấn luyện trực tiếp trên
Market-1501. So sánh với nó sẽ phản ánh sai kết quả.

Biến thể AIN — phiên bản tổng quát hoá miền của cùng nhóm tác giả — vượt mô hình
gốc 3,6 điểm rank-1, nên được chọn làm mặc định. Điều này đúng với mục đích
thiết kế của AIN: triển khai trên những camera mô hình chưa từng thấy, tức đúng
tình huống của dự án này.

### Mức 2 — toàn hệ thống, đầu-đến-cuối

Các danh tính Market-1501 được đưa vào hệ thống dưới dạng tracklet và truy vấn
qua API thật, nên mức này đo cả việc chọn crop, nhúng trung bình nhiều crop,
ngưỡng và kiểm tra thời gian — không chỉ riêng mô hình.

Tinh chỉnh trên **`bounding_box_train` (751 danh tính)**, tách biệt hoàn toàn
với các danh tính `query`/`bounding_box_test` dùng để báo cáo ở Mức 1. 150 truy
vấn xuyên camera được giữ lại, đối chiếu với 1.315 tracklet từ 400 danh tính.

| τ (ngưỡng) | Top-1 | Top-5 | Kết quả sai trả về |
|---|---|---|---|
| 0,50 | 69,3% | **85,3%** | 467 |
| 0,65 | 69,3% | 85,3% | 457 |
| 0,70 | 69,3% | 84,0% | 403 |
| 0,74 | 68,0% | 81,3% | 314 |
| 0,78 | 60,0% | 68,7% | 205 |
| 0,82 | 48,0% | 54,0% | 94 |
| 0,90 | 10,0% | 11,3% | 5 |

**Một ghi chú phương pháp đáng lưu lại.** Phiên bản đánh giá trước đó tinh
chỉnh τ *và* chọn mô hình trên cùng tập dữ liệu dùng để báo cáo. Kết quả là
top-1 đạt 100% — trông rất tốt và hoàn toàn vô nghĩa. Sau khi chuyển việc tinh
chỉnh sang các danh tính tách biệt, con số trung thực là **69,3%**. Khoảng chênh
lệch đó chính là mức độ mà rò rỉ tập kiểm tra (test-set leak) làm đẹp kết quả.
Script sinh bảng hiện tự đánh dấu "LEAKED" nếu hai tập dữ liệu lại trùng nhau.

**Lựa chọn ngưỡng.** τ được đặt là **0,50**, ưu tiên độ phủ (recall). Top-5
giảm từ 85,3% xuống 81,3% khi τ tăng từ 0,50 lên 0,74, trong khi số kết quả sai
giảm từ 467 xuống 314. Với bài toán này, cái giá hai bên không đối xứng: người
kiểm duyệt dễ dàng bỏ qua vài ảnh thu nhỏ sai, nhưng một lần xuất hiện bị bỏ sót
nghĩa là hệ thống báo rằng người mất tích chưa từng ở đó. Recall là chỉ số phù
hợp với mục đích của sản phẩm.

### Mức 3 — độ trễ trực tiếp (R8)

| Phép đo | Giá trị |
|---|---|
| Số lần xuất hiện đo được | 17 |
| Trung vị thời gian đến khi tìm được | **2,13 s** |
| Trường hợp xấu nhất | 2,18 s |
| Yêu cầu | ≤ 10 s |

Đo từ thời điểm người rời khỏi khung hình đến khi lần xuất hiện của họ có thể
tìm thấy. Các lần chạy trước dưới tải cao hơn, gồm cả hai worker đồng thời, cho
trung vị 2,2–2,5 s và xấu nhất 3,9 s — vẫn nằm sâu trong yêu cầu.

Độ trễ bị chi phối bởi thời gian chờ mất dấu 2 giây, không phải bởi năng lực
tính toán. Chạy trên GPU thay vì CPU gần như không thay đổi (trung vị 2,12 s so
với 2,24 s), nghĩa là GPU mang lại khả năng xử lý thêm camera hoặc độ phân giải
cao hơn, chứ không phải độ trễ thấp hơn.

### Thông lượng

| Cấu hình | Khung hình/giây |
|---|---|
| CPU, 768×432 | 10,9 |
| GPU (ROCm), 768×432 | 28,5 |
| GPU, 720p, một worker | 14,1 |
| GPU, 720p, hai worker | 11,4 + 11,5 = 22,9 tổng cộng |

Hai tiến trình worker cho tổng thông lượng cao hơn một tiến trình, nghĩa là một
worker chưa làm GPU đạt tải tối đa. Hai camera ở 10 fps cần tổng 20 fps, còn
khoảng 15% dư.

## 5. Tín hiệu có thực sự hoạt động?

Trước khi xây dựng hệ thống, vector nhúng OSNet được đo trực tiếp trên các crop
đã theo vết để kiểm chứng tiền đề cốt lõi:

| | Độ tương đồng cosine |
|---|---|
| Cùng một người, hai nửa của một vết | 0,791 – 0,903 (trung bình 0,842) |
| Các vết khác nhau | cao nhất 0,873 (trung bình 0,720) |

Độ tương đồng của cùng một người rõ ràng cao hơn của những người khác nhau, xác
nhận hướng tiếp cận. Phép đo này cũng **phủ định một tham số đã dự kiến**: bản
đặc tả kỹ thuật đề xuất khởi đầu với τ = 0,5 như một giá trị an toàn, nhưng mọi
cặp quan sát được đều có điểm trên 0,5, nên ngưỡng đó sẽ không lọc được gì cả.
Dải giá trị thực tế là 0,6–0,9.

## 6. Quyền riêng tư và đạo đức

Thiết kế coi quyền riêng tư là một ràng buộc, không phải phần bổ sung sau cùng.

- **Tìm kiếm bắt đầu từ một ảnh.** Không có cách nào để xem danh sách tất cả
  những người đã đi qua; hệ thống không có khái niệm danh tính, chỉ có vector
  ngoại hình.
- **Không gán nhãn cho người không khớp.** Những người được phát hiện nhưng
  không khớp không bao giờ bị đặt tên, gắn thẻ hay liên kết giữa các lần tìm.
- **Con người ra quyết định.** Mọi kết quả chỉ là gợi ý kèm điểm số. Người kiểm
  duyệt xác nhận hoặc từ chối từng kết quả; hệ thống không bao giờ khẳng định.
- **Thời hạn lưu trữ.** Video, lần xuất hiện và clip tự xoá sau 3 ngày. Ảnh
  tham chiếu bị xoá khi đóng phiên tìm kiếm. Dữ liệu đánh giá được đánh dấu
  bảo vệ để không bị tác vụ xoá phá huỷ tập đã gán nhãn.
- **Chỉ chạy cục bộ.** Toàn bộ xử lý diễn ra trên một máy. Không video hay ảnh
  nào được gửi tới dịch vụ bên ngoài.
- **Thiên lệch đã biết.** Độ chính xác giảm khi thay đổi quần áo, ánh sáng và
  góc camera. Vì việc so khớp dựa nhiều vào quần áo, đổi áo khoác sẽ làm hệ
  thống thất bại — điều này được nêu rõ ở đây thay vì che giấu, vì nó giới hạn
  những gì hệ thống có thể khẳng định một cách trung thực.

## 7. Hạn chế

Nêu thẳng, vì chúng giới hạn ý nghĩa của các kết quả.

**Mục tiêu top-1 ≥ 80% không đạt được.** Con số trung thực là 69,3% trên các
danh tính Market-1501. Kết quả 100% trước đó là do rò rỉ tập kiểm tra, không
phải thành tích.

**Chưa hoàn thành đánh giá trên dữ liệu tự quay.** Việc quay tình nguyện viên
tại hai vị trí thực tế không diễn ra trong thời gian cho phép, nên mọi số liệu
độ chính xác đều đến từ bộ dữ liệu công khai. Việc các con số này có chuyển
được sang một hệ camera cụ thể hay không vẫn chưa được kiểm chứng — và đó chính
là điểm yếu nhất của re-ID.

**Chỉ một đối tượng.** Vì không có tình nguyện viên, không thể đo mức độ nhầm
lẫn giữa các danh tính trên camera của chính dự án. Bộ gây nhiễu Market-1501 là
phương án thay thế hợp lý, nhưng crop của nó sạch kiểu dữ liệu nghiên cứu, chứ
không phải được lấy qua pipeline này.

**Không xây dựng so khớp khuôn mặt (R12).** Ở khoảng cách camera giám sát, khuôn
mặt chỉ chiếm 20–40 pixel, thấp hơn nhiều mức ~112×112 mà nhận diện khuôn mặt
cần. Do đó hệ thống không phù hợp cho định danh ở khoảng cách gần, nơi nhận diện
khuôn mặt mới là công cụ đúng.

**Chưa kiểm chứng phần cứng.** Script cho node camera Raspberry Pi và luồng SRT
từ iPhone đã được viết nhưng chưa bao giờ chạy với thiết bị thật. Kiểm thử trực
tiếp dùng webcam USB và luồng phát lại từ tệp.

**Tìm kiếm văn bản phụ thuộc ngưỡng.** Độ tương đồng văn bản–ảnh của CLIP thay
đổi theo câu truy vấn (kết quả tốt nhất 0,25–0,35 so với trung vị 0,17–0,23),
nên một ngưỡng cố định là mong manh. Một quy tắc tương đối — lấy top-k kèm sàn
tối thiểu — sẽ bền hơn.

## 8. Nếu làm lại

**Đo trước khi thiết kế quanh một con số.** Giá trị τ = 0,5 trong đặc tả kỹ
thuật là một phỏng đoán bị một thí nghiệm 20 phút phủ định. Nhiều quyết định về
sau sẽ rẻ hơn nếu phép đo đó được thực hiện trước.

**Không bao giờ tinh chỉnh trên tập dùng để báo cáo.** Rò rỉ tạo ra kết quả
100% hoàn toàn không thể thấy được từ đầu ra — các con số chỉ đơn giản trông
đẹp. Chỉ khi tách hai tập ra thì vấn đề mới lộ. Mọi đánh giá đáng công bố cần
sự tách biệt đó ngay từ đầu, không phải chắp vá về sau.

**Quay dữ liệu sớm.** Dữ liệu tình nguyện viên là đường găng cho mọi con số trên
dữ liệu tự quay, và là phụ thuộc duy nhất không thể song song hoá hay thay thế.
Nó nên được xếp làm việc đầu tiên, không phải cuối cùng.

## 9. Kết luận

Hệ thống làm được điều nó đặt ra: một ảnh thu hẹp một tập lớn người được phát
hiện xuống một danh sách ngắn đã xếp hạng mà người kiểm duyệt có thể xem trong
vài phút, trên cả tệp ghi sẵn và camera trực tiếp, với lần xuất hiện tìm được
khoảng 2 giây sau khi người đó rời khung hình — tốt hơn yêu cầu 10 giây khoảng
năm lần.

Mục tiêu độ chính xác không đạt. Top-1 69,3% nghĩa là khoảng một trong ba lần
tìm kiếm không đặt đúng người ở vị trí đầu, dù top-5 85,3% nghĩa là họ thường
vẫn nằm trong trang kết quả đầu tiên — và với một công cụ có nhiệm vụ thu hẹp
phạm vi tìm kiếm chứ không phải tự quyết định, đó là con số đáng quan tâm hơn.

Kết quả giá trị nhất không phải là hệ thống, mà là kỷ luật đo lường: một ngưỡng
lẽ ra không lọc được gì, một lựa chọn mô hình bị đảo ngược khi kiểm tra công
bằng, và một con số độ chính xác rơi từ 100% xuống 69% sau khi loại bỏ rò rỉ.
Không điều nào trong số đó lộ ra trước khi được đo đúng cách.
