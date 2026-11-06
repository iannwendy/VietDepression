# Hướng dẫn review nhãn bằng Label Studio

Tài liệu này dành cho người được giao việc **review thủ công** comment tiếng Việt để gán nhãn trầm cảm. Đọc hết một lượt trước khi bắt đầu.

---

## 0. Bối cảnh ngắn (đọc 1 phút, quan trọng)

Hệ thống đã tự gán nhãn sơ bộ bằng từ khóa và bằng mô hình PhoBERT. Việc của bạn là **gán nhãn lại bằng con người** cho một số mẫu, để có "nhãn vàng" (gold) đáng tin.

> **Một lỗi đã xảy ra trước đây và phải tránh:** ở lần review trước, người review nhìn thấy gợi ý của máy rồi bấm đồng ý theo, nên 96% nhãn người trùng với máy. Kết quả là điểm đánh giá mô hình bị **ảo (chính xác 100%)** và vô dụng.
>
> Vì vậy file bạn nhận **đã ẩn toàn bộ gợi ý của máy**. Bạn chỉ thấy câu comment và phải tự quyết định. Đây là chủ đích, không phải thiếu sót.

Bạn sẽ review **2 file**, ứng với 2 đợt:

| Đợt | File import | Số dòng | Ưu tiên |
|---|---|---|---|
| Đợt 1 | `label_studio_step8_active_learning_import.csv` | 1.000 | **Làm trước** (mẫu mô hình thấy khó nhất, giá trị cao nhất) |
| Đợt 2 | `label_studio_step5_review_import.csv` | 750 | Làm sau |

> Chỉ import file có đuôi `_import.csv`. **Không** đụng tới file `_key.csv` — đó là file hệ thống giữ để ghép nhãn về sau.

---

## 1. Cài đặt Label Studio

Label Studio là công cụ gán nhãn dữ liệu mã nguồn mở, chạy local trên máy bạn.

### Cách A — Dùng Python (khuyến nghị)

Yêu cầu: Python 3.8–3.11 đã cài sẵn.

```bash
# 1. Tạo môi trường riêng để không đụng tới project
python3 -m venv ls-venv
source ls-venv/bin/activate        # Windows: ls-venv\Scripts\activate

# 2. Cài Label Studio
pip install label-studio

# 3. Khởi động
label-studio start
```

Trình duyệt sẽ tự mở tại `http://localhost:8080`. Lần đầu vào, đăng ký một tài khoản local bất kỳ (email + mật khẩu, chỉ lưu trên máy bạn).

### Cách B — Dùng Docker

Yêu cầu: Docker Desktop đang chạy.

```bash
docker run -it -p 8080:8080 -v $(pwd)/ls-data:/label-studio/data heartexlabs/label-studio:latest
```

Mở `http://localhost:8080`.

> Nếu cổng 8080 bị chiếm, đổi sang cổng khác: `label-studio start -p 8090` hoặc `-p 8090:8080` với Docker.

---

## 2. Tạo project và import dữ liệu

1. Đăng nhập → bấm **Create Project**.
2. Tab **Project Name**: đặt tên, ví dụ `Review Dot 1 - Active Learning`.
3. Tab **Data Import**: bấm **Upload Files**, chọn file `label_studio_step8_active_learning_import.csv`.
   - Khi hỏi *"Treat CSV as..."* → chọn **List of tasks**.
4. Tab **Labeling Setup**: chọn **Custom template** rồi dán cấu hình ở mục 3 bên dưới.
5. Bấm **Save**.

Làm lại y hệt cho file đợt 2 trong một project riêng (`Review Dot 2`).

---

## 3. Cấu hình giao diện gán nhãn (Labeling Config)

Vào **Settings → Labeling Interface → Code**, xóa hết và dán đoạn này:

```xml
<View>
  <Header value="Đọc kỹ câu dưới đây rồi chọn nhãn. Tự quyết định, KHÔNG có gợi ý của máy."/>

  <Text name="comment" value="$text"/>

  <Choices name="final_label" toName="comment" choice="single" required="true" showInline="false">
    <Choice value="depression" hotkey="1"/>
    <Choice value="normal" hotkey="2"/>
    <Choice value="uncertain" hotkey="3"/>
    <Choice value="exclude" hotkey="4"/>
  </Choices>

  <Header value="Ghi chú (bắt buộc khi chọn uncertain hoặc khi bạn phân vân):"/>
  <TextArea name="reviewer_note" toName="comment" rows="2" maxSubmissions="1" editable="true"/>
</View>
```

Giải thích:
- `$text` sẽ hiển thị nội dung comment.
- 4 nhãn có phím tắt **1/2/3/4** để gán nhanh.
- Ô ghi chú để bạn ghi lý do khi phân vân.

Bấm **Save**.

---

## 4. Tiêu chí gán nhãn (phần quan trọng nhất)

Đọc câu comment và tự hỏi: **người viết câu này có đang bộc lộ tín hiệu trầm cảm / đau khổ tâm lý cá nhân không?**

### `depression` — có tín hiệu trầm cảm/đau khổ cá nhân
Người viết đang nói về **chính họ**, thể hiện một trong các dấu hiệu:
- Buồn bã / tuyệt vọng / trống rỗng kéo dài.
- Mất động lực, mệt mỏi tinh thần, kiệt sức, mất ngủ vì tâm lý.
- Cô đơn, cảm thấy vô dụng, ghét bản thân, không ai hiểu.
- Ý nghĩ tự hại, muốn biến mất, không thiết sống.
- Nhắc tới trầm cảm/lo âu của bản thân.

Ví dụ → `depression`:
- *"Dạo này mình mệt mỏi quá, chẳng muốn làm gì, đêm nào cũng khóc một mình."*
- *"Tôi thấy mình vô dụng, sống chẳng để làm gì."*

### `normal` — không có tín hiệu trầm cảm
- Bình luận khen/chê, hỏi đáp, bàn luận chuyện ngoài.
- Buồn **thoáng qua** vì chuyện nhỏ, không phải đau khổ kéo dài.
- Tức giận / chê bai về sản phẩm, dịch vụ, người khác.
- **Câu động viên người khác** (đây là bẫy hay gặp — xem dưới).

Ví dụ → `normal`:
- *"Video hay quá, cảm ơn bạn nhiều!"*
- *"Đồ ăn quán này dở, không đáng tiền."*
- *"Cố lên bạn nhé, đừng nghĩ quẩn, mọi chuyện sẽ ổn thôi."* ← người viết đang **an ủi người khác**, bản thân họ không bị → `normal`.

### `uncertain` — không đủ thông tin để quyết
- Câu quá mơ hồ, thiếu ngữ cảnh, hiểu kiểu nào cũng được.
- Bạn đọc 2–3 lần vẫn không chắc.
- **Bắt buộc ghi lý do** vào ô ghi chú.

Ví dụ → `uncertain`:
- *"Lại một ngày nữa..."* (không rõ buồn hay chỉ kể chuyện)

### `exclude` — loại bỏ, không dùng được
- Không phải tiếng Việt, hoặc chỉ toàn emoji/ký tự.
- Spam, quảng cáo, link.
- Quá ngắn và vô nghĩa: *"haha"*, *"k"*, *"=))"*.

---

## 5. Bốn nguyên tắc vàng khi review

1. **Đọc câu TRƯỚC, quyết định RỒI mới sang câu sau.** Không tìm kiếm gợi ý — file đã ẩn hết rồi, cứ tin vào phán đoán của mình.

2. **Phân biệt "người viết bị" và "người viết nói về người khác / chủ đề".** Một video tựa đề "dấu hiệu trầm cảm" có thể nhận hàng trăm comment **bình thường** ("video bổ ích ạ"). Chủ đề là trầm cảm ≠ người bình luận bị trầm cảm.

3. **Câu khuyên/an ủi người khác → `normal`.** "Đừng tự tử nhé", "hãy đi khám tâm lý đi bạn" là người viết đang giúp người khác, không phải họ đang đau khổ.

4. **Bạn SẼ bất đồng với máy ở một số câu — điều đó là tốt.** Nếu bạn thấy mình chọn `depression`/`normal` đều tăm tắp như nhau cho mọi câu thì hãy chậm lại. Các mẫu trong file đợt 1 cố tình là mẫu **khó** mà máy phân vân; tỉ lệ nhãn lệch nhau là dấu hiệu bạn đang đọc thật.

---

## 6. Gán nhãn

- Trong project, bấm vào task đầu tiên → đọc câu → bấm phím **1/2/3/4** (hoặc click) → bấm **Submit**.
- Task tiếp theo tự hiện. Lặp lại.
- Khi chọn `uncertain`, nhớ gõ lý do vào ô ghi chú trước khi Submit.
- Có thể dừng bất cứ lúc nào; tiến độ được lưu tự động. Lần sau quay lại làm tiếp.

Mục tiêu: cố gắng hoàn thành **cả 1.000 mẫu đợt 1** trước, rồi tới 750 mẫu đợt 2.

---

## 7. Export kết quả và bàn giao

Khi xong một project:

1. Vào project → bấm **Export**.
2. Chọn định dạng **CSV**.
3. Tải file về.
4. **Đổi tên file** rõ ràng rồi gửi lại cho người phụ trách pipeline:
   - Đợt 1 → `export_step8_active_learning.csv`
   - Đợt 2 → `export_step5_review.csv`

> File export sẽ có cột `row_id`, `final_label`, `reviewer_note`. Hệ thống dùng `row_id` để ghép nhãn người của bạn về đúng dòng gốc (qua file `_key.csv`). **Đừng sửa cột `row_id`.**

Bàn giao xong là hết phần việc của bạn. Cảm ơn — phần review này là bước quyết định chất lượng của cả mô hình.

---

## 8. Lưu ý đạo đức

- Dữ liệu là comment công khai, **đã ẩn tên người dùng**.
- Đây là gán nhãn **tín hiệu ngôn ngữ** phục vụ nghiên cứu, **không phải chẩn đoán y tế** cho bất kỳ ai.
- Không chia sẻ nội dung comment ra ngoài nhóm nghiên cứu.
- Nếu nội dung khiến bạn thấy nặng nề, cứ tạm nghỉ. Sức khỏe của bạn quan trọng hơn tiến độ.
