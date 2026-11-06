# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=577 ; `class_name`=gong ; `rank`=1 ; `score`=0.420191 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   cả bức ảnh. Vì không thấy các có thông tin vị trí như x, y, width, height
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   Danh sách lớp không phải do model tự nghĩ ra mà nó được xác định bởi bộ dữ liệu và taxonomy dùng để huấn luyện mô hình, sau đó được liên kết với checkpoint
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   Hai taxonomy khác nhau có thể sử dụng cùng một tên lớp nhưng có cách định nghĩa hoặc phân nhóm khác nhau, class_id giúp hệ thống xác định chính xác lớp trong danh sách lớp của taxonomy tương ứng
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   Chọn vật thể hoặc chủ đề chiếm vai trò trung tâm trong ảnh. Xem xét kích thước, vị trí, độ rõ nét và mức độ thu hút sự chú ý. Xem xét nội dung tổng thể của ảnh, không chỉ dựa vào một chi tiết nhỏ. Nếu có nhiều vật thể nổi bật, phải quy định vật thể hoặc chủ đề nào được ưu tiên
6. Vì sao không được lấy `score` của model làm ground truth?
   Vì score là mức độ tin cậy hoặc điểm dự đoán của model, không phải nhãn thực tế đã được xác minh

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   Góc trên bên-trái nằm ở (385.33pixel, 69.24pixel), góc dưới-phải nằm ở (498.92pixel, 348.92pixel), rộng 113.58 pixel, cao 279.68 pixel
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   chọn ngưỡng 0.35 và 0.60. Ngưỡng 0.35 có 11 prediction và 0.60 có 6 prediction
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   khi hạ ngưỡng người reviewer cần làm kiểm tra từng bounding box có thực sự bao quanh vật thể hay không, kiểm tra class có đúng với vật thể được phát hiện không, phát hiện các bounding box bị sai vị trí, quá rộng hoặc quá hẹp, kiểm tra các box trùng lặp trên cùng một vật thể
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   Box phải bám sát đường viền vật thể. Không bao gồm vùng nền không thuộc vật thể. Nếu vật thể bị che khuất, chỉ bao quanh phần thực sự nhìn thấy. Nếu vật thể bị cắt bởi mép ảnh, box kết thúc tại mép ảnh. Mỗi vật thể được gán một box riêng, không gộp nhiều vật thể cùng lớp vào một box.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   vật thể bị che khuất hoặc cắt mép là person 0.61. Khi vật thể bị cắt bởi mép ảnh, người gán nhãn chỉ vẽ bounding box bao quanh phần vật thể còn nhìn thấy trong ảnh. Không mở rộng bounding box ra ngoài kích thước ảnh và không tự suy đoán phần vật thể bị khuất.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-006 ; `class_name`=bowl ; `score`=0.528201 ; số điểm polygon=`polygon_point_count`=84 ; 3–5 điểm đầu của `polygon_xy`=[
   29.0,
   303.0
   ],
   [
   28.0,
   304.0
   ],
   [
   26.0,
   304.0
   ],
   [
   25.0,
   305.0
   ],
   [
   23.0,
   305.0
   ],
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   box chỉ có khung còn polygon cho thêm các đường viền bôi vật thể, box chỉ có 4 góc còn polygon cho thêm nhiều điểm tọa độ, box có chứa các vùng thừa nhiều hơn còn polygon bo tròn các vật thể hạn chế nhiều vùng thừa
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   `instance_id` là mã định danh của một đối tượng cụ thể trong ảnh. class ID dùng để xác định vật thể, tracking ID dùng để theo dõi 1 vật thể qua nhiều khung hình, Instance ID dùng để phân biệt vật thể riêng biệt trong ảnh
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   Đặt các điểm polygon dọc theo đường viền vật thể. Không bao gồm vùng nền bên ngoài. Không gộp hai vật thể riêng biệt thành một mask. Khi vật thể bị che khuất, chỉ vẽ phần nhìn thấy được. Đóng polygon để tạo thành một vùng mask hoàn chỉnh
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   khu vực mặt bàn bên trái, nơi các chiếc đĩa/bát và phần bột hoặc thực phẩm đặt sát nhau. Nên sử dụng escalation vì không xác định được biên

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON     | Annotator làm gì?                     | Reviewer xem gì?                                       |
| --------------------- | --------------------------------------------- | ------------------------------------------------------ | ------------------------------------- | ------------------------------------------------------ |
| Phân loại ảnh         | Một class cho toàn ảnh                        | Model dự đoán gong, score 0.420191                     | Chọn class đúng theo ảnh và guideline | Kiểm tra nhãn có đúng nội dung ảnh                     |
| Phát hiện vật thể     | Class + bounding box                          | Người bị cắt mép trái; box có thể sai hoặc trùng       | Vẽ box sát vật thể, gán đúng class    | Kiểm tra class, tọa độ, box trùng và vật thể bị bỏ sót |
| Instance segmentation | Class + instance ID + polygon                 | Bát gần mặt bàn, biên khó xác định; polygon có 84 điểm | Vẽ mask sát biên vật thể              | Kiểm tra biên mask, vùng nền thừa và instance bị gộp   |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   Không tự ý sao chép, chia sẻ hoặc sử dụng dữ liệu lab ngoài phạm vi được cho phép
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   Dừng xử lý và báo cho Lab Coach hoặc Project Lead để được hướng dẫn

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
