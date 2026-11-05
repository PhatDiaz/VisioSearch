# Hướng dẫn chạy VisioSearch (Tiếng Việt)

## Phiên bản hiện tại — SigLIP LoRA + BLIP ITM

Nhánh `feature/two-stage-retrieval` đã dùng mặc định:
**SigLIP LoRA `batch16_r8_e5` → Top-20 trên 8.000 ảnh → BLIP ITM → fusion alpha=0.75**.
Giao diện, logo, sáng/tối, gallery và lựa chọn 5/10/20 ảnh được giữ nguyên.
Điểm trên thẻ ảnh là **match score** sau fusion; CSV và popup còn giữ cosine/logit gốc.

Giữ nguyên thư mục ảnh `data/flickr8k`, đặt đủ ba artifacts đã cung cấp vào
`research_artifacts`, rồi chạy một lần trong terminal PowerShell của VS Code:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/prepare_two_stage.py --download-models
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Khi đã kích hoạt môi trường Python, lệnh chạy vẫn là `python -m streamlit run app.py`.
Các lần sau chỉ cần `start.bat` hoặc lệnh Streamlit; không giải nén lại ZIP khi mở app.
Model/embeddings runtime được lưu dưới `models/` và tên file LoRA mới trong `data/flickr8k`.
Script kiểm tra fingerprint của metadata, adapter và vectors; không thay ảnh, metadata,
embeddings pretrained hay kết quả nghiên cứu cũ. Nếu thiếu artifacts/model sẽ báo lỗi,
không tự đổi về pretrained SigLIP.

**Đo thật trên CPU hiện tại:** lượt đầu 14,83 giây (SigLIP 5,38; BLIP 9,44),
một câu mới khi model đã nạp 9,33 giây (SigLIP 0,05; BLIP 9,28).
Truy vấn giống hệt dùng lại cache. Đây là thời gian pipeline, chưa tính hiển thị browser.
BLIP chấm 20 cặp ảnh-văn bản nên chậm hơn phiên bản pretrained một giai đoạn.
CUDA được tự phát hiện khi dùng bản PyTorch hỗ trợ CUDA; lần kiểm tra này chạy CPU.

**Kết quả nghiên cứu 63,14 / 83,08 / 88,06%** là số trong ZIP đã cung cấp.
Pipeline tích hợp chưa chạy lại đủ 5.000 truy vấn để xác nhận các con số này.
Ba truy vấn mẫu đã khớp Top-1 lưu sẵn; không được dùng mẫu đó thay cho benchmark đầy đủ.

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_two_stage.py --reference-queries 3
```

Chi tiết công thức, test, CUDA/batch size và cách đánh giá đủ 5.000 truy vấn nằm ở đầu README.md.
Các hướng dẫn phía dưới chủ yếu mô tả chuẩn bị Flickr8k và dashboard baseline v1.

## Chạy nhanh trên máy Windows này

Dữ liệu đã xuất từ Colab nằm trong `Topic11_8K_Outputs`. Sau khi sao chép
`Topic11_8K_Outputs/data/flickr8k` vào `data/flickr8k` và
`Topic11_8K_Outputs/results` vào `results`, không cần xây dựng lại chỉ mục.

Khi đã cài thư viện vào `.venv`, nhấp đúp `start.bat`, hoặc chạy:

```powershell
.\start.ps1
```

Mở **http://localhost:8501**. Nhấn `Ctrl+C` trong cửa sổ chạy app để dừng.
Script dùng Python của `.venv`, nên không cần lệnh `python` trong PATH hoặc
kích hoạt môi trường thủ công. Model tải lần đầu được lưu trong
`.cache/huggingface` của project; các lần chạy sau dùng lại cache này.

## Mục tiêu

- Tìm kiếm bằng câu tiếng Anh trên **8.000 ảnh Flickr8k**.
- Trang chính dùng **SigLIP**, cho chọn 5/10/20 kết quả và xem ảnh trong gallery responsive.
- Nút **View image** mở ảnh gốc, metadata và captions; có thể tải ảnh gốc hoặc xuất kết quả CSV.
- Dashboard nghiên cứu riêng giữ CLIP/SigLIP, Recall@K, failure cases và methodology.

**Trạng thái baseline v1 đã kiểm tra trước đây:** Đã nhập đủ 8.000 ảnh và embeddings từ
`Topic11_8K_Outputs`, tải hai model, chạy tìm kiếm thật bằng CPU và kiểm tra giao diện.
Cả 11 tests đều qua. Kết quả Recall có sẵn đã được đối chiếu với file rank của
từng truy vấn; chưa chạy lại toàn bộ indexing hoặc benchmark 5.000 truy vấn trên máy này.

**Tốc độ baseline pretrained v1:** Model đã tải được ưu tiên đọc trực tiếp từ cache, không kiểm
tra Hugging Face qua mạng. Điều này tránh chờ thử lại nhiều lần khi mạng bị chặn.
Đo kiểm tra nội bộ trên CPU: lượt SigLIP đầu khoảng 5 giây để nạp model; các câu
mới tiếp theo khoảng 0,2 giây. Thời gian hiển thị trên trình duyệt có thể khác.

## Cách A — Máy có GPU phù hợp

1. Cài Python 3.11 và giải nén project.
2. Mở Terminal ngay tại thư mục `Topic11_8K_Search_App`.
3. Lần lượt chạy:

```bash
python -m venv .venv
source .venv/bin/activate  # Mac / Linux
pip install -r requirements.txt
python scripts/build_gallery.py --output-dir data/flickr8k
python scripts/evaluate.py --index-dir data/flickr8k
streamlit run app.py
```

Trên Windows PowerShell thay dòng kích hoạt bằng `.venv\Scripts\Activate.ps1`.

4. Mở địa chỉ **http://localhost:8501**. Nhập mô tả tiếng Anh rồi nhấn **Search** hoặc Enter.

## Cách B — Laptop không có GPU (đề xuất)

1. Giải nén file project nhưng **chưa cần chạy indexing trên laptop**.
2. Upload notebook `notebooks/Topic11_Prepare_8000_Gallery.ipynb` vào **Google Colab** và bật GPU.
3. Trong notebook, upload file ZIP mã nguồn của project khi được yêu cầu.
4. Chạy các cell để tải 8.000 ảnh, xây dựng CLIP / SigLIP embeddings và tính test metrics.
5. Chuyển archive `.tar.gz` chứa `data/flickr8k` và `results` về laptop (ưu tiên qua Google Drive nếu file lớn).
6. Giải nén archive **ngay trong thư mục project**.
7. Cài môi trường trên laptop (3 lệnh đầu như trên), sau đó chạy `streamlit run app.py`.

Colab chỉ dùng làm máy **chuẩn bị dữ liệu và chạy mô hình**. Người dùng tương tác với giao diện riêng trên laptop, không cần nhìn thấy notebook.

## Khi app chạy

- Trang chính chỉ gồm ô tìm kiếm và ảnh kết quả, không có tab nghiên cứu hoặc nút chọn model.
- Chọn 5, 10 hoặc 20 kết quả, nhấn Enter hoặc **Search**. Các gợi ý bên dưới cũng bấm được.
- **View image:** xem ảnh gốc, captions, kích thước và ID ảnh.
- **Export results:** tải bảng kết quả CSV của truy vấn hiện tại.

Để mở dashboard học thuật, chạy một terminal khác:

```powershell
.\.venv\Scripts\python.exe -m streamlit run research_app.py --server.port 8502
```

Mở **http://localhost:8502** để xem lại Model evaluation, Failure explorer,
Methodology và so sánh CLIP/SigLIP. Toàn bộ kết quả đánh giá và cấu trúc dữ liệu được giữ nguyên.

## Điều tuyệt đối không làm khi báo cáo

- Không lấy kết quả trên 200 ảnh validation gán là kết quả trên 8.000 ảnh.
- Không lấy 40.000 captions của toàn bộ dữ liệu rồi gọi là test độc lập.
- Không hiển thị điểm cosine similarity như xác suất % ảnh đúng.
- Với baseline v1, không mô tả pretrained zero-shot là fine-tuning. Với pipeline hiện tại,
  ghi rõ sử dụng adapter LoRA đã huấn luyện trong nghiên cứu; lần tích hợp app này không huấn luyện thêm.

## Sửa lỗi 429 trên Google Colab

Nếu notebook cũ báo `HTTPError: 429 Too Many Requests` khi `Loading Flickr8k split: train`, đây là lỗi bị giới hạn số request tải ảnh từng file, **không phải lỗi CLIP/SigLIP**. Các lỗi `metadata.jsonl` và `benchmark_results.csv` bị thiếu chỉ là hậu quả.

**Cách sửa:** sử dụng ZIP mã nguồn và notebook **đã cập nhật**. Script mới tải ảnh và caption từ các file Parquet đã đóng gói của **chính** `intro/flickr8k` (revision `refs/convert/parquet`) thay vì tải 8.000 file ảnh riêng. Chạy tuần tự các cell. Notebook kiểm tra số lượng ảnh, embedding và chỉ chạy đánh giá khi dữ liệu đầy đủ.

- Cell 09 sẽ tạo và tải `Topic11_Evaluation_Results.zip` nhỏ để gửi kiểm tra.
- Cell 10 tạo `Topic11_8K_Outputs.tar.gz` lớn để mang về máy chạy website.
- Cell 11 có thể copy file lớn sang Google Drive (`EXPORT_TO_DRIVE = True`).

Nếu Hugging Face vẫn báo 429 với các file Parquet hoặc model checkpoint, dừng các lượt tải liên tục và chạy lại cell lỗi khi giới hạn dịch vụ đã được gỡ.
